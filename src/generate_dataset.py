#!/usr/bin/env python3
"""
Generates a synthetic tool-calling dataset for an Indian GST assistant.

Every tool result is computed by real Python code (not by an LLM), so the
ground truth is exact. Run:  python generate_dataset.py --n 3000 --out data
"""
import argparse, calendar, hashlib, json, random, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

SYSTEM = (
    "You are GSTBuddy, an assistant for Indian GST questions and calculations. "
    "Use the provided tools for any calculation, lookup or validation instead of doing arithmetic yourself. "
    "If required details are missing, ask one short clarifying question. "
    "You are not a tax professional; for complex matters recommend a Chartered Accountant."
)

# ----------------------------------------------------------------------------
# Reference data (rates reflect GST 2.0, effective 22 Sep 2025; illustrative subset)
# ----------------------------------------------------------------------------
RATES = {
    "shampoo": (5, ["shampoo", "hair shampoo"]),
    "toothpaste": (5, ["toothpaste"]),
    "butter": (5, ["butter"]),
    "ghee": (5, ["ghee", "desi ghee"]),
    "namkeen": (5, ["namkeen", "bhujia"]),
    "tractor": (5, ["tractor", "tractors"]),
    "drip irrigation system": (5, ["drip irrigation system", "drip irrigation"]),
    "thermometer": (5, ["thermometer", "digital thermometer"]),
    "solar panel": (5, ["solar panel", "solar panels"]),
    "bicycle": (5, ["bicycle", "bicycles"]),
    "air conditioner": (18, ["air conditioner", "ac", "split ac"]),
    "dishwasher": (18, ["dishwasher"]),
    "cement": (18, ["cement"]),
    "smartphone": (18, ["smartphone", "mobile phone", "phone"]),
    "structural steel": (18, ["structural steel"]),
    "small car": (18, ["small car", "hatchback"]),
    "motorcycle up to 350cc": (18, ["motorcycle up to 350cc", "bike under 350cc"]),
    "coal": (18, ["coal"]),
    "aerated drink": (40, ["aerated drink", "cola", "soft drink"]),
    "pan masala": (40, ["pan masala"]),
    "cigarettes": (40, ["cigarettes", "cigarette"]),
    "motorcycle above 350cc": (40, ["motorcycle above 350cc", "bike above 350cc"]),
    "individual health insurance": (0, ["individual health insurance", "health insurance policy"]),
    "notebook": (0, ["notebook", "notebooks", "exercise book"]),
}
UNKNOWN_ITEMS = ["wooden furniture", "mobile phone cover", "LED bulb", "school bag",
                 "hair dryer", "ayurvedic hair tonic", "printer cartridge", "yoga mat"]
RATE_NOTE = "Reference table follows GST 2.0 (effective 22 Sep 2025). Confirm the HSN-specific rate in the latest notification."

STATES = {"Maharashtra": "27", "Karnataka": "29", "Gujarat": "24", "Tamil Nadu": "33", "Delhi": "07",
          "Uttar Pradesh": "09", "West Bengal": "19", "Rajasthan": "08", "Telangana": "36",
          "Kerala": "32", "Punjab": "03", "Haryana": "06", "Madhya Pradesh": "23", "Bihar": "10", "Goa": "30"}
STATE_BY_CODE = {v: k for k, v in STATES.items()}
CH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ----------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ----------------------------------------------------------------------------
def _tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}

TOOL_SCHEMAS = [
    _tool("calculate_gst", "Calculate GST on an amount, either added on top (exclusive) or extracted from a GST-inclusive price.",
          {"amount": {"type": "number", "description": "Amount in INR"},
           "rate": {"type": "number", "description": "GST rate in percent, e.g. 5, 18, 40"},
           "mode": {"type": "string", "enum": ["exclusive", "inclusive"], "description": "Whether amount excludes or includes GST"}},
          ["amount", "rate", "mode"]),
    _tool("split_gst", "Split a GST amount into CGST+SGST (intra-state) or IGST (inter-state).",
          {"gst_amount": {"type": "number"}, "supply_type": {"type": "string", "enum": ["intra_state", "inter_state"]}},
          ["gst_amount", "supply_type"]),
    _tool("determine_supply_type", "Decide whether a supply is intra-state or inter-state from seller and buyer (place of supply) states.",
          {"seller_state": {"type": "string"}, "buyer_state": {"type": "string"}}, ["seller_state", "buyer_state"]),
    _tool("get_gst_rate", "Look up the GST rate for a goods/service item from the reference table.",
          {"item": {"type": "string", "description": "Item name in lowercase, e.g. 'shampoo'"}}, ["item"]),
    _tool("validate_gstin", "Validate the format and checksum of a 15-character GSTIN. Does not confirm the GSTIN is registered.",
          {"gstin": {"type": "string"}}, ["gstin"]),
    _tool("calculate_interest", "Interest on delayed GST payment at 18% per annum (simple interest, days/365).",
          {"tax_amount": {"type": "number"}, "days_late": {"type": "integer"}}, ["tax_amount", "days_late"]),
    _tool("get_filing_due_date", "Standard due date of GSTR-1 (11th) or GSTR-3B (20th) for a tax period, for regular monthly filers.",
          {"return_type": {"type": "string", "enum": ["GSTR-1", "GSTR-3B"]},
           "month": {"type": "integer", "description": "Tax period month 1-12"},
           "year": {"type": "integer", "description": "Tax period year"}}, ["return_type", "month", "year"]),
    _tool("calculate_net_gst_payable", "Net GST payable after set-off of input tax credit (ITC) against output tax.",
          {"output_tax": {"type": "number"}, "input_tax_credit": {"type": "number"}}, ["output_tax", "input_tax_credit"]),
]

# ----------------------------------------------------------------------------
# Tool implementations (ground truth)
# ----------------------------------------------------------------------------
def r2(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def t_calculate_gst(amount, rate, mode):
    a, r = Decimal(str(amount)), Decimal(str(rate))
    if mode == "exclusive":
        base, gst = a, a * r / 100
        total = base + gst
    else:
        total = a
        base = a * 100 / (100 + r)
        gst = total - base
    return {"base_amount": r2(base), "gst_amount": r2(gst), "total_amount": r2(total), "rate": rate, "mode": mode}

def t_split_gst(gst_amount, supply_type):
    if supply_type == "intra_state":
        c = r2(Decimal(str(gst_amount)) / 2)
        return {"cgst": c, "sgst": r2(Decimal(str(gst_amount)) - Decimal(str(c))), "igst": 0.0}
    return {"cgst": 0.0, "sgst": 0.0, "igst": r2(gst_amount)}

def t_determine_supply_type(seller_state, buyer_state):
    same = seller_state.strip().lower() == buyer_state.strip().lower()
    return {"supply_type": "intra_state" if same else "inter_state",
            "tax_applicable": "CGST+SGST" if same else "IGST"}

def t_get_gst_rate(item):
    q = item.strip().lower()
    for canon, (rate, aliases) in RATES.items():
        if q == canon or q in aliases:
            return {"found": True, "item": canon, "rate": rate, "note": RATE_NOTE}
    return {"found": False, "item": q, "note": "Item not in reference table. Ask for the HSN/SAC code or check the GST rate notification."}

def gstin_check_char(first14):
    s = 0
    for i, c in enumerate(first14):
        p = CH.index(c) * (1 if i % 2 == 0 else 2)
        s += p // 36 + p % 36
    return CH[(36 - s % 36) % 36]

def t_validate_gstin(gstin):
    g = gstin.strip().upper()
    if len(g) != 15:
        return {"valid": False, "reason": "invalid_length", "state": None}
    if not re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]", g):
        return {"valid": False, "reason": "invalid_format", "state": None}
    state = STATE_BY_CODE.get(g[:2])
    if g[:2] not in STATE_BY_CODE and not (1 <= int(g[:2]) <= 38):
        return {"valid": False, "reason": "invalid_state_code", "state": None}
    if gstin_check_char(g[:14]) != g[14]:
        return {"valid": False, "reason": "checksum_mismatch", "state": state}
    return {"valid": True, "reason": "ok", "state": state, "pan": g[2:12]}

def t_calculate_interest(tax_amount, days_late):
    i = Decimal(str(tax_amount)) * Decimal("0.18") * days_late / 365
    return {"interest": r2(i), "rate_percent_per_annum": 18, "days_late": days_late}

def t_get_filing_due_date(return_type, month, year):
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    day = 11 if return_type == "GSTR-1" else 20
    return {"return_type": return_type, "tax_period": f"{year}-{month:02d}",
            "due_date": f"{ny}-{nm:02d}-{day:02d}", "note": "Standard date for regular monthly filers; may change with government extensions."}

def t_calculate_net_gst_payable(output_tax, input_tax_credit):
    net = Decimal(str(output_tax)) - Decimal(str(input_tax_credit))
    if net >= 0:
        return {"net_payable": r2(net), "carry_forward_itc": 0.0}
    return {"net_payable": 0.0, "carry_forward_itc": r2(-net)}

TOOLS = {n[2:]: f for n, f in list(globals().items()) if n.startswith("t_") and callable(f)}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def indian(n):
    whole, _, frac = f"{Decimal(str(n)):.2f}".partition(".")
    if len(whole) > 3:
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", whole[:-3])
        whole = head + "," + whole[-3:]
    return whole if frac == "00" else f"{whole}.{frac}"

def rs(n):
    return "₹" + indian(n)

def money(a, rng):
    s = indian(a)
    return rng.choice([f"₹{s}", f"Rs. {s}", f"{s} rupees", f"₹{s}", f"{str(a).rstrip('0').rstrip('.') if '.' in str(a) else a} INR"])

def amount(rng):
    a = rng.choice([rng.randint(100, 5000), rng.randint(5000, 100000), rng.randint(100000, 2500000)])
    return a + rng.choice([0, 0, 0, 0.5, 0.25]) if a < 100000 else a

class Conv:
    def __init__(self, user):
        self.m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        self.n = 0
        self.gold = []

    def call(self, calls):
        tcs, outs = [], []
        for name, args in calls:
            self.n += 1
            tcs.append({"id": f"call_{self.n}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
            outs.append(TOOLS[name](**args))
            self.gold.append({"name": name, "arguments": args})
        self.m.append({"role": "assistant", "content": None, "tool_calls": tcs})
        for tc, o in zip(tcs, outs):
            self.m.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"],
                           "content": json.dumps(o, ensure_ascii=False)})
        return outs

    def say(self, text):
        self.m.append({"role": "assistant", "content": text})
        return self

def pick(rng, train, unseen, p=0.12):
    if unseen and rng.random() < p:
        return rng.choice(unseen), True
    return rng.choice(train), False

def pick_lang(rng):
    return rng.choices(["en", "hi"], [0.65, 0.35])[0]  # "hi" = Hinglish (Roman script)

# ----------------------------------------------------------------------------
# Generators: each returns (Conv, category, lang, unseen_template)
# ----------------------------------------------------------------------------
def g_calc(rng):
    lang, mode, rate, a = pick_lang(rng), rng.choice(["exclusive", "inclusive"]), rng.choice([5, 18, 18, 40, 5]), amount(rng)
    A = money(a, rng)
    T = {
        ("en", "exclusive"): (["Calculate {rate}% GST on {A}.", "My invoice value is {A} before tax. What is the total with {rate}% GST?",
                               "How much GST will I charge on {A} at {rate}%?", "Add {rate}% GST to {A} and give me the final amount."],
                              ["Need the GST component and final price for a {A} sale (pre-tax) at {rate} percent."]),
        ("en", "inclusive"): (["{A} is the GST-inclusive price at {rate}%. What are the base price and GST?",
                               "Extract GST from {A}, which already includes {rate}% tax.",
                               "What's the pre-tax amount if the final price is {A} with {rate}% GST included?"],
                              ["Customer paid {A} all-in, rate {rate}%. Break it up for the invoice."]),
        ("hi", "exclusive"): (["{A} ka {rate}% GST kitna banega?", "Mera bill {A} hai GST ke bina, {rate}% lagake total batao.",
                               "{A} pe {rate}% GST add karke final amount bata do."],
                              ["Bhai {A} ki sale hai (tax alag se), {rate} percent pe GST aur total kya aayega?"]),
        ("hi", "inclusive"): (["{A} GST included hai {rate}% pe, base amount aur GST alag karo.", "{A} mein {rate}% GST shamil hai, GST kitna hai nikalo."],
                              ["Customer ne {A} diye total, {rate}% rate hai, invoice ke liye breakup do."]),
    }[(lang, mode)]
    tpl, unseen = pick(rng, *T)
    c = Conv(tpl.format(A=A, rate=rate))
    out = c.call([("calculate_gst", {"amount": a, "rate": rate, "mode": mode})])[0]
    b, g, t = rs(out["base_amount"]), rs(out["gst_amount"]), rs(out["total_amount"])
    if mode == "exclusive":
        c.say(f"GST at {rate}% on {b} is {g}, so the total invoice value is {t}." if lang == "en"
              else f"{b} pe {rate}% GST {g} banta hai, to total invoice value {t} hogi.")
    else:
        c.say(f"Out of {t}, the base (pre-tax) amount is {b} and the GST at {rate}% is {g}." if lang == "en"
              else f"{t} mein base amount {b} hai aur {rate}% GST {g} hai.")
    return c, "calculate_gst", lang, unseen

def g_split(rng):
    lang, st, g = pick_lang(rng), rng.choice(["intra_state", "inter_state"]), amount(rng)
    G = money(g, rng)
    kind = {"intra_state": ("within the same state", "ek hi state ke andar"), "inter_state": ("to a customer in another state", "doosre state ke customer ko")}[st]
    tpl, unseen = pick(rng, [f"Split {G} of GST for a sale {kind[0]}.", f"I charged {G} GST on a sale {kind[0]}. How is it divided?"] if lang == "en"
                       else [f"{G} GST ko split karo, sale {kind[1]} hai.", f"{kind[1]} sale pe {G} GST laga hai, CGST SGST IGST kaise bantega?"],
                       [f"Break {G} GST into CGST/SGST/IGST, sale {kind[0]}."] if lang == "en" else [f"{G} ke GST ka breakup do, sale {kind[1]} hui."])
    c = Conv(tpl)
    o = c.call([("split_gst", {"gst_amount": g, "supply_type": st})])[0]
    if st == "intra_state":
        c.say(f"For an intra-state supply, the {G} GST splits equally into CGST {rs(o['cgst'])} and SGST {rs(o['sgst'])}." if lang == "en"
              else f"Intra-state supply mein {G} GST barabar bantta hai: CGST {rs(o['cgst'])} aur SGST {rs(o['sgst'])}.")
    else:
        c.say(f"For an inter-state supply, the full {G} is charged as IGST." if lang == "en" else f"Inter-state supply mein poora {G} IGST ke roop mein lagta hai.")
    return c, "split_gst", lang, unseen

def g_supply(rng):
    lang = pick_lang(rng)
    s1, s2 = rng.choice(list(STATES)), rng.choice(list(STATES))
    if rng.random() < 0.4:
        s2 = s1
    tpl, unseen = pick(rng, ["Seller is in {s1} and buyer is in {s2}. Is it CGST+SGST or IGST?", "Goods go from {s1} to {s2}. Which GST applies?"] if lang == "en"
                       else ["Seller {s1} mein hai aur buyer {s2} mein. CGST SGST lagega ya IGST?", "{s1} se {s2} maal ja raha hai, kaunsa GST lagega?"],
                       ["Supplier based in {s1}, delivering to a client in {s2}: intra or inter-state?"] if lang == "en" else ["{s1} ka supplier, {s2} ka client. Intra-state hai ya inter-state?"])
    c = Conv(tpl.format(s1=s1, s2=s2))
    o = c.call([("determine_supply_type", {"seller_state": s1, "buyer_state": s2})])[0]
    if o["supply_type"] == "intra_state":
        c.say(f"Both are in {s1}, so it is an intra-state supply and CGST + SGST apply." if lang == "en" else f"Dono {s1} mein hain, isliye intra-state supply hai aur CGST + SGST lagega.")
    else:
        c.say(f"{s1} to {s2} is an inter-state supply, so IGST applies." if lang == "en" else f"{s1} se {s2} inter-state supply hai, isliye IGST lagega.")
    return c, "determine_supply_type", lang, unseen

def g_rate(rng):
    lang = pick_lang(rng)
    canon = rng.choice(list(RATES))
    phrase = rng.choice(RATES[canon][1])
    tpl, unseen = pick(rng, ["What is the GST rate on {i}?", "How much GST is charged on {i}?", "Tell me the GST slab for {i}."] if lang == "en"
                       else ["{i} pe kitna GST lagta hai?", "{i} ka GST rate kya hai?"],
                       ["Which tax rate applies if I sell {i}?"] if lang == "en" else ["{i} bechne pe GST kitna dena padega?"])
    c = Conv(tpl.format(i=phrase))
    o = c.call([("get_gst_rate", {"item": phrase})])[0]
    c.say(f"{phrase.capitalize()} attracts {o['rate']}% GST under the rates effective since 22 September 2025. Please confirm against the latest notification for your exact HSN code." if lang == "en"
          else f"{phrase.capitalize()} pe 22 September 2025 se lagu rates ke hisaab se {o['rate']}% GST lagta hai. Apne HSN code ke liye latest notification zaroor check kar lena.")
    return c, "get_gst_rate", lang, unseen

def g_unknown_item(rng):
    lang, item = pick_lang(rng), rng.choice(UNKNOWN_ITEMS)
    c = Conv(rng.choice([f"What GST rate applies to {item}?"] if lang == "en" else [f"{item} pe kitna GST hai?"]))
    c.call([("get_gst_rate", {"item": item.lower()})])
    c.say(f"I couldn't find {item} in my reference table, so I can't give a reliable rate. Please share its HSN code or check the official GST rate notification, or ask a CA." if lang == "en"
          else f"{item} mere reference table mein nahi mila, isliye pakka rate nahi bata sakta. HSN code share karo ya official GST rate notification / CA se confirm kar lo.")
    return c, "get_gst_rate_unknown", lang, False

def make_gstin(rng, valid=True):
    sc = rng.choice(list(STATE_BY_CODE))
    pan = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3)) + rng.choice("CPFHTA") + rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") \
          + "".join(rng.choice("0123456789") for _ in range(4)) + rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    first14 = sc + pan + rng.choice("123456789") + "Z"
    g = first14 + gstin_check_char(first14)
    if not valid:
        mode = rng.choice(["checksum", "checksum", "length", "state"])
        if mode == "checksum":
            g = first14 + rng.choice([x for x in CH if x != g[14]])
        elif mode == "length":
            g = g[:-1] if rng.random() < 0.5 else g + rng.choice("0123456789")
        else:
            g = "99" + g[2:]
    return g

def g_gstin(rng):
    lang, valid = pick_lang(rng), rng.random() < 0.5
    g = make_gstin(rng, valid)
    tpl, unseen = pick(rng, ["Is {g} a valid GSTIN?", "Please check this GSTIN: {g}", "Validate GSTIN {g}"] if lang == "en"
                       else ["Kya {g} sahi GSTIN hai?", "Ye GSTIN check karo: {g}"],
                       ["My vendor gave me {g} on the invoice. Does it look right?"] if lang == "en" else ["Vendor ne invoice pe {g} diya hai, ye theek lag raha hai?"])
    c = Conv(tpl.format(g=g))
    o = c.call([("validate_gstin", {"gstin": g})])[0]
    reasons = {"invalid_length": "it is not 15 characters long", "invalid_format": "its format is wrong",
               "invalid_state_code": "the state code is invalid", "checksum_mismatch": "the checksum digit doesn't match"}
    reasons_hi = {"invalid_length": "ye 15 characters ka nahi hai", "invalid_format": "iska format galat hai",
                  "invalid_state_code": "state code galat hai", "checksum_mismatch": "checksum digit match nahi karta"}
    if o["valid"]:
        c.say(f"{g} passes the format and checksum checks (state: {o['state'] or 'valid state code'}). Note this doesn't confirm it is actually registered; verify on the GST portal." if lang == "en"
              else f"{g} format aur checksum check pass karta hai (state: {o['state'] or 'valid state code'}). Ye confirm nahi karta ki registered hai; GST portal pe verify karo.")
    else:
        c.say(f"{g} is not valid: {reasons[o['reason']]}. Please double-check it with your vendor." if lang == "en"
              else f"{g} valid nahi hai: {reasons_hi[o['reason']]}. Vendor se dobara confirm kar lo.")
    return c, "validate_gstin", lang, unseen

def g_interest(rng):
    lang, tax, d = pick_lang(rng), amount(rng), rng.randint(1, 365)
    T = money(tax, rng)
    tpl, unseen = pick(rng, ["I paid {T} of GST {d} days late. How much interest do I owe?", "Interest on {T} GST delayed by {d} days?"] if lang == "en"
                       else ["{T} ka GST {d} din late bhara, interest kitna lagega?", "{T} GST {d} din deri se, interest batao."],
                       ["Compute the penalty interest for {T} tax paid {d} days after the due date."] if lang == "en" else ["{T} tax due date ke {d} din baad bhara, interest nikalo."])
    c = Conv(tpl.format(T=T, d=d))
    o = c.call([("calculate_interest", {"tax_amount": tax, "days_late": d})])[0]
    c.say(f"Interest at 18% per annum on {T} for {d} days comes to {rs(o['interest'])}." if lang == "en"
          else f"{T} pe {d} din ka 18% p.a. interest {rs(o['interest'])} banta hai.")
    return c, "calculate_interest", lang, unseen

def g_due(rng):
    lang, rt, m, y = pick_lang(rng), rng.choice(["GSTR-1", "GSTR-3B"]), rng.randint(1, 12), rng.choice([2025, 2026, 2027])
    mn = calendar.month_name[m]
    tpl, unseen = pick(rng, ["When is {rt} due for {mn} {y}?", "What's the due date of {rt} for the {mn} {y} tax period?"] if lang == "en"
                       else ["{mn} {y} ka {rt} kab tak bharna hai?", "{rt} ki due date kya hai {mn} {y} ke liye?"],
                       ["Deadline to file {rt} for {mn} {y}?"] if lang == "en" else ["{mn} {y} ke {rt} ki last date?"])
    c = Conv(tpl.format(rt=rt, mn=mn, y=y))
    o = c.call([("get_filing_due_date", {"return_type": rt, "month": m, "year": y})])[0]
    yy, mm, dd = map(int, o["due_date"].split("-"))
    ds = f"{dd} {calendar.month_name[mm]} {yy}"
    c.say(f"{rt} for {mn} {y} is due on {ds} (standard date for regular monthly filers). It can shift if the government extends the deadline, so check the GST portal." if lang == "en"
          else f"{mn} {y} ka {rt} {ds} tak bharna hai (regular monthly filers ke liye standard date). Extension aa sakta hai, to GST portal check karte rehna.")
    return c, "get_filing_due_date", lang, unseen

def g_net(rng):
    lang, out_t = pick_lang(rng), amount(rng)
    itc = int(out_t * rng.choice([0.3, 0.6, 0.9, 1.0, 1.2, 1.5]))
    O, I = money(out_t, rng), money(itc, rng)
    tpl, unseen = pick(rng, ["My output tax is {O} and my ITC is {I}. How much GST do I pay?", "Output GST {O}, input tax credit {I}. Net payable?"] if lang == "en"
                       else ["Output tax {O} hai aur ITC {I}. Kitna GST bharna padega?", "Meri sales ka GST {O}, purchase ka ITC {I}. Net kitna?"],
                       ["After setting off {I} of ITC against {O} output tax, what remains?"] if lang == "en" else ["{O} output tax mein se {I} ITC minus karke kitna bachega?"])
    c = Conv(tpl.format(O=O, I=I))
    o = c.call([("calculate_net_gst_payable", {"output_tax": out_t, "input_tax_credit": itc})])[0]
    if o["net_payable"] > 0:
        c.say(f"After using {I} of ITC, you need to pay {rs(o['net_payable'])} of GST." if lang == "en" else f"{I} ITC adjust karne ke baad aapko {rs(o['net_payable'])} GST bharna hoga.")
    else:
        c.say(f"Your ITC covers the full output tax, so nothing is payable now and {rs(o['carry_forward_itc'])} of ITC carries forward." if lang == "en"
              else f"ITC se poora output tax cover ho gaya, abhi kuch nahi bharna aur {rs(o['carry_forward_itc'])} ITC aage carry forward hoga.")
    return c, "calculate_net_gst_payable", lang, unseen

def g_multi_rate_calc(rng):
    lang, canon, a = pick_lang(rng), rng.choice(list(RATES)), None
    a = amount(rng)
    phrase, A = rng.choice(RATES[canon][1]), money(a, rng)
    tpl, unseen = pick(rng, ["I'm selling {i} for {A} excluding GST. What will the total be?", "Price of my {i} is {A} before tax. Final invoice amount?"] if lang == "en"
                       else ["{i} {A} mein bech raha hoon (GST alag), total kya hoga?", "{i} ka price {A} hai tax se pehle, final invoice amount batao."],
                       ["Quote the tax-inclusive price for a {i} priced at {A} net."] if lang == "en" else ["{i} ka net price {A} hai, tax ke saath quote kitna banega?"])
    c = Conv(tpl.format(i=phrase, A=A))
    rate = c.call([("get_gst_rate", {"item": phrase})])[0]["rate"]
    o = c.call([("calculate_gst", {"amount": a, "rate": rate, "mode": "exclusive"})])[0]
    c.say(f"{phrase.capitalize()} is taxed at {rate}%, so GST on {A} is {rs(o['gst_amount'])} and the total comes to {rs(o['total_amount'])}. Confirm the rate for your HSN code before invoicing." if lang == "en"
          else f"{phrase.capitalize()} pe {rate}% GST lagta hai, to {A} pe GST {rs(o['gst_amount'])} aur total {rs(o['total_amount'])} hoga. Invoice se pehle apne HSN ka rate confirm kar lena.")
    return c, "multi_rate_then_calc", lang, unseen

def g_multi_invoice(rng):
    lang, a, rate = pick_lang(rng), amount(rng), rng.choice([5, 18, 18, 40])
    s1, s2 = rng.choice(list(STATES)), rng.choice(list(STATES))
    if rng.random() < 0.4:
        s2 = s1
    A = money(a, rng)
    tpl, unseen = pick(rng, ["Seller in {s1}, buyer in {s2}, invoice {A} before tax at {rate}% GST. Show the CGST/SGST/IGST breakup.",
                             "Make the tax breakup for a {A} sale (excl. GST) at {rate}% from {s1} to {s2}."] if lang == "en"
                       else ["Seller {s1} mein, buyer {s2} mein, invoice {A} tax se pehle, {rate}% GST. CGST/SGST/IGST breakup do.",
                             "{s1} se {s2} ko {A} ki sale (GST alag) {rate}% pe, tax breakup banao."],
                       ["Invoice total and tax split needed: {A} net, {rate}% GST, {s1} -> {s2}."] if lang == "en" else ["{s1} -> {s2}, {A} net, {rate}% GST; invoice total aur tax split chahiye."])
    c = Conv(tpl.format(s1=s1, s2=s2, A=A, rate=rate))
    sup, calc = c.call([("determine_supply_type", {"seller_state": s1, "buyer_state": s2}),
                        ("calculate_gst", {"amount": a, "rate": rate, "mode": "exclusive"})])
    sp = c.call([("split_gst", {"gst_amount": calc["gst_amount"], "supply_type": sup["supply_type"]})])[0]
    tot = rs(calc["total_amount"])
    if sup["supply_type"] == "intra_state":
        c.say(f"Both parties are in {s1}, so this is intra-state. On {A} at {rate}%: CGST {rs(sp['cgst'])} + SGST {rs(sp['sgst'])}. Invoice total: {tot}." if lang == "en"
              else f"Dono {s1} mein hain, to intra-state hai. {A} pe {rate}%: CGST {rs(sp['cgst'])} + SGST {rs(sp['sgst'])}. Invoice total: {tot}.")
    else:
        c.say(f"{s1} to {s2} is inter-state, so IGST applies. On {A} at {rate}%: IGST {rs(sp['igst'])}. Invoice total: {tot}." if lang == "en"
              else f"{s1} se {s2} inter-state hai, to IGST lagega. {A} pe {rate}%: IGST {rs(sp['igst'])}. Invoice total: {tot}.")
    return c, "multi_invoice_breakup", lang, unseen

CLARIFY = [
    ("Calculate GST on {A}.", "Sure. What GST rate applies, and is {A} inclusive or exclusive of GST?", "{A} pe GST nikalna hai.", "Zaroor. Kaunsa GST rate lagta hai, aur {A} GST ke saath hai ya bina?"),
    ("Is this sale CGST/SGST or IGST?", "Which state is the seller in, and which state is the buyer in?", "Ye sale CGST/SGST wali hai ya IGST?", "Seller kis state mein hai aur buyer kis state mein?"),
    ("How much interest do I owe on late GST payment?", "How much tax was due, and how many days late was the payment?", "Late GST payment pe interest kitna lagega?", "Kitna tax due tha aur payment kitne din late hui?"),
    ("Is my GSTIN valid?", "Please share the 15-character GSTIN you want me to check.", "Mera GSTIN sahi hai kya?", "Jo 15-character GSTIN check karna hai wo share karo."),
    ("When is my GST return due?", "Which return (GSTR-1 or GSTR-3B) and for which month and year?", "Mera GST return kab bharna hai?", "Kaunsa return (GSTR-1 ya GSTR-3B), aur kis month-year ka?"),
]

def g_clarify(rng):
    lang, row = pick_lang(rng), rng.choice(CLARIFY)
    A = money(amount(rng), rng)
    q, a = (row[0], row[1]) if lang == "en" else (row[2], row[3])
    return Conv(q.format(A=A)).say(a.format(A=A)), "clarify", lang, False

CONCEPTS = [
    (["What is the difference between CGST, SGST and IGST?", "Explain CGST vs SGST vs IGST."],
     "CGST and SGST are charged together on intra-state supplies, with the tax split equally between the Centre (CGST) and the state (SGST). IGST is charged on inter-state supplies; it is collected by the Centre and the destination state's share is passed on.",
     ["CGST, SGST aur IGST mein kya fark hai?"],
     "Intra-state supply mein CGST + SGST lagta hai, jisme tax Centre (CGST) aur state (SGST) mein barabar bantta hai. Inter-state supply mein IGST lagta hai, jo Centre collect karta hai aur destination state ka hissa usse transfer hota hai."),
    (["What is input tax credit (ITC)?", "Explain ITC in simple words."],
     "Input tax credit lets a registered business reduce the GST it has paid on purchases from the GST it owes on sales, subject to conditions such as holding a valid tax invoice, receiving the goods or services, and the supplier having paid the tax.",
     ["ITC kya hota hai?"],
     "Input tax credit ka matlab hai ki purchases pe diya hua GST aap apni sales ke GST se minus kar sakte ho, kuch conditions ke saath, jaise valid tax invoice, maal ya service mil gayi ho, aur supplier ne tax jama kiya ho."),
    (["What is the difference between GSTR-1 and GSTR-3B?", "GSTR-1 vs GSTR-3B?"],
     "GSTR-1 reports your outward supplies (sales) invoice-wise. GSTR-3B is the summary return where you declare your tax liability, claim ITC and pay the net tax.",
     ["GSTR-1 aur GSTR-3B mein kya difference hai?"],
     "GSTR-1 mein aapki outward supplies (sales) invoice-wise report hoti hain. GSTR-3B summary return hai jisme tax liability declare hoti hai, ITC claim hota hai aur net tax bhara jata hai."),
    (["What is an HSN code?", "What does HSN stand for in GST?"],
     "HSN (Harmonised System of Nomenclature) is a standard code that classifies goods, and the GST rate is tied to it. Services use SAC codes instead.",
     [], None),
    (["What are the current GST slabs in India?", "Which GST rates are applicable now?"],
     "Since the GST 2.0 reforms took effect on 22 September 2025, the main rates are 5% and 18%, plus a 40% rate for select luxury and sin goods. 0% applies to nil-rated or exempt items, and special rates such as 3% on gold continue. The old 12% and 28% slabs were removed. The exact rate depends on the HSN code, so check the latest notification before invoicing.",
     [], None),
    (["What is reverse charge in GST?", "Explain reverse charge mechanism."],
     "Under reverse charge, the recipient of the supply pays the GST directly to the government instead of the supplier. It applies to specified goods and services and certain supplies from unregistered suppliers.",
     [], None),
]

PRE = ["", "", "Quick question: ", "Hey, ", "Hi GSTBuddy, ", "Sir, ", "Please tell me: "]
SUF = ["", "", " Thanks!", " Keep it short.", " Explain simply."]

def decorate(q, rng):
    return rng.choice(PRE) + q + rng.choice(SUF)

def g_concept(rng):
    qs, a, qh, ah = rng.choice(CONCEPTS)
    if ah and rng.random() < 0.35:
        return Conv(decorate(rng.choice(qh), rng)).say(ah), "concept", "hi", False
    return Conv(decorate(rng.choice(qs), rng)).say(a), "concept", "en", False

OOS = [
    ("Write me a poem about the monsoon.", "I'm focused on GST questions and calculations, so I can't help with that. If you have a GST query, I'm happy to help."),
    ("Which stock should I buy this week?", "That's outside what I can help with. I stick to GST questions and calculations. Ask me about rates, invoices, or returns anytime."),
    ("How can I avoid paying GST by not giving bills to customers?", "I can't help with evading tax. Legitimate options worth discussing with a Chartered Accountant include the composition scheme if you're eligible, claiming input tax credit correctly, and checking whether your supplies are exempt or nil-rated."),
    ("What's the best biryani place in Pune?", "I can't help with restaurant recommendations. I only handle GST questions and calculations, so ask me about rates, invoices, ITC or returns."),
    ("Can you write Python code to sort a list?", "That's outside my scope, as I focus on GST questions and calculations. If you need a GST calculation or lookup, I'm ready."),
    ("What is the income tax slab for FY 2026-27?", "I only cover GST, not income tax, so I can't give a reliable answer on that. A Chartered Accountant or the Income Tax Department's website is the right place. I'm happy to help with any GST question."),
    ("Tell me a joke.", "I'd rather stick to GST. Ask me about rates, invoices, ITC or return due dates and I'll help."),
    ("Which GST number should I put on the invoice if I don't have one? Just make one up.", "I can't make up a GSTIN. If you're not registered, the invoice shouldn't show one. If you need to register, a CA or the GST portal can guide you."),
    ("Make a fake GST invoice for me so I can claim ITC.", "I can't help create a fake invoice, since claiming ITC on fabricated invoices is a serious offence. I can help you work out ITC on genuine purchases or prepare a correct tax invoice."),
]

def g_oos(rng):
    q, a = rng.choice(OOS)
    return Conv(decorate(q, rng)).say(a), "out_of_scope", "en", False

GENS = [(g_calc, 0.14), (g_split, 0.07), (g_supply, 0.07), (g_rate, 0.08), (g_unknown_item, 0.03), (g_gstin, 0.09), (g_interest, 0.07),
        (g_due, 0.07), (g_net, 0.06), (g_multi_rate_calc, 0.10), (g_multi_invoice, 0.10), (g_clarify, 0.05), (g_concept, 0.05), (g_oos, 0.04)]

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    seen, rows = set(), []
    for gen, w in GENS:
        target, tries = max(1, round(a.n * w)), 0
        got = 0
        while got < target and tries < target * 30:
            tries += 1
            conv, cat, lang, unseen = gen(rng)
            key = conv.m[1]["content"] + json.dumps(conv.gold)
            if key in seen:
                continue
            seen.add(key); got += 1
            rows.append({"id": hashlib.md5(key.encode()).hexdigest()[:12], "category": cat, "lang": lang,
                         "unseen_template": unseen, "gold_calls": json.dumps(conv.gold, ensure_ascii=False), "messages": conv.m})

    rng.shuffle(rows)
    train, test = [], []
    for r in rows:
        (test if r["unseen_template"] or int(r["id"], 16) % 10 == 0 else train).append(r)

    for name, data in (("train", train), ("test", test)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out.parent / "tools.json").write_text(json.dumps(TOOL_SCHEMAS, indent=2), encoding="utf-8")

    from collections import Counter
    print(f"train={len(train)} test={len(test)} (unseen-template test rows: {sum(r['unseen_template'] for r in test)})")
    print("categories:", dict(Counter(r["category"] for r in rows)))
    print("languages:", dict(Counter(r["lang"] for r in rows)))

if __name__ == "__main__":
    main()
