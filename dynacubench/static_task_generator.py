"""
Generate additional static-baseline tasks for DynaCU-Bench v3.

The static baseline (Category S) verifies the AOI's "scales to zero" claim:
on tasks with no dynamic content, the AOI must not degrade performance.

Initial benchmark had 10 easy tasks (S-E1..S-E10).
This module adds 40 more — 12 easy, 16 medium, 12 hard — for a total of 50.

All tasks are pure HTML/CSS:
  - no animations, transitions, video, audio, timers
  - the page is a still image: every visible element is the same at t=0 and t=∞
  - DOM-based success check via window.getTaskResult()
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

OUT_DIR = Path("/home/ubuntu/adaptive-observation-paper/benchmark_env/html_tasks")

PAGE_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#f6f7fb; color:#1a1a2e; font-family:'Segoe UI',Tahoma,sans-serif; min-height:100vh; padding:32px 20px; display:flex; justify-content:center; }}
  .card {{ background:#fff; border-radius:12px; box-shadow:0 2px 14px rgba(0,0,0,.08); max-width:760px; width:100%; padding:28px 32px; }}
  h1 {{ font-size:24px; margin-bottom:16px; color:#0f172a; }}
  h2 {{ font-size:18px; margin:20px 0 10px; color:#334155; }}
  .instructions {{ background:#eef2ff; border-left:4px solid #4f46e5; padding:12px 16px; margin-bottom:18px; font-size:14px; color:#3730a3; line-height:1.55; border-radius:0 8px 8px 0; }}
  label {{ display:block; font-size:13px; font-weight:600; color:#475569; margin:10px 0 4px; }}
  input[type=text], input[type=email], input[type=number], select, textarea {{ width:100%; padding:10px 14px; font-size:15px; border:2px solid #e2e8f0; border-radius:7px; outline:none; }}
  input:focus, select:focus, textarea:focus {{ border-color:#4f46e5; }}
  button {{ margin-top:14px; padding:11px 26px; background:#4f46e5; color:#fff; border:none; border-radius:7px; font-size:15px; font-weight:600; cursor:pointer; }}
  button:hover {{ background:#4338ca; }}
  table {{ width:100%; border-collapse:collapse; margin:8px 0; font-size:14px; }}
  th, td {{ padding:8px 12px; border-bottom:1px solid #e2e8f0; text-align:left; }}
  th {{ background:#f1f5f9; }}
  .muted {{ color:#64748b; font-size:13px; }}
  .pill {{ display:inline-block; padding:2px 10px; border-radius:99px; background:#e0e7ff; color:#3730a3; font-size:12px; font-weight:600; }}
  .row {{ display:flex; gap:16px; }}
  .row > * {{ flex:1; }}
  ul {{ padding-left:20px; }}
  li {{ margin:4px 0; }}
</style>
</head>
<body>
<div class="card">
"""

PAGE_FOOTER = """
</div>
<script>{script}</script>
</body>
</html>
"""


def emit(filename, title, body_html, success_js):
    html = (
        PAGE_HEADER.format(title=title)
        + body_html
        + PAGE_FOOTER.format(script=success_js)
    )
    (OUT_DIR / filename).write_text(html)


# ── EASY tasks (12 new) ────────────────────────────────────────────────

def make_easy():
    # S-E11: Read a phone number from a contact card
    emit(
        "S_E11_read_phone.html",
        "Read Phone Number",
        '<h1>Customer Contact Card</h1>'
        '<p class="muted">Look up the contact information below.</p>'
        '<div style="background:#f8fafc;padding:16px;border-radius:8px;margin:14px 0">'
        '<p><strong>Name:</strong> Robert Tanaka</p>'
        '<p><strong>Department:</strong> Engineering</p>'
        '<p><strong>Phone:</strong> (415) 555-0182</p>'
        '<p><strong>Email:</strong> r.tanaka@example.com</p></div>'
        '<label>Type the phone number exactly as shown:</label>'
        '<input type="text" id="ans" autocomplete="off">'
        '<button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{}; '
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim();'
        ' return v==="(415) 555-0182"?"phone_correct":"incorrect";};'
    )

    # S-E12: Pick a category radio button
    emit(
        "S_E12_radio_category.html",
        "Select Category",
        '<h1>Support Ticket</h1>'
        '<p class="instructions">Choose the most relevant category for your issue.</p>'
        '<p>I have a question about an invoice that was charged twice last month.</p>'
        '<div style="margin:14px 0">'
        '<label><input type="radio" name="cat" value="general"> General inquiry</label>'
        '<label><input type="radio" name="cat" value="technical"> Technical support</label>'
        '<label><input type="radio" name="cat" value="billing" id="r_billing"> Billing</label>'
        '<label><input type="radio" name="cat" value="feedback"> Feedback</label></div>'
        '<button id="submitBtn">Submit</button>',
        'window.__chosen=null;'
        'document.getElementById("submitBtn").onclick=()=>{const r=document.querySelector("input[name=cat]:checked"); window.__chosen=r?r.value:null;};'
        'window.getTaskResult=()=>window.__chosen==="billing"?"category_correct":"incorrect";'
    )

    # S-E13: Read a numeric value from a settings panel
    emit(
        "S_E13_read_setting.html",
        "Read Setting Value",
        '<h1>Display Settings</h1>'
        '<table><tr><th>Setting</th><th>Value</th></tr>'
        '<tr><td>Brightness</td><td>72%</td></tr>'
        '<tr><td>Contrast</td><td>50%</td></tr>'
        '<tr><td>Refresh rate</td><td>144 Hz</td></tr>'
        '<tr><td>Resolution</td><td>2560×1440</td></tr></table>'
        '<label>What is the refresh rate? (number only, e.g. 60)</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim();'
        ' return v==="144"?"refresh_correct":"incorrect";};'
    )

    # S-E14: Sum two visible prices
    emit(
        "S_E14_sum_prices.html",
        "Sum Prices",
        '<h1>Order Summary</h1>'
        '<table><tr><th>Item</th><th>Price</th></tr>'
        '<tr><td>USB-C Cable</td><td>$12.50</td></tr>'
        '<tr><td>Laptop Stand</td><td>$37.25</td></tr></table>'
        '<label>Total in dollars (e.g. 49.75):</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim();'
        ' return ["49.75","$49.75","49.75 dollars"].includes(v.toLowerCase())?"sum_correct":"incorrect";};'
    )

    # S-E15: Choose multi-select toppings
    emit(
        "S_E15_multi_select.html",
        "Multi-Select",
        '<h1>Pizza Order</h1>'
        '<p>Select the toppings: <strong>mushroom, olive, pepper</strong> (no others).</p>'
        '<div>'
        '<label><input type="checkbox" id="t_mushroom"> Mushroom</label>'
        '<label><input type="checkbox" id="t_pepperoni"> Pepperoni</label>'
        '<label><input type="checkbox" id="t_olive"> Olive</label>'
        '<label><input type="checkbox" id="t_pepper"> Pepper</label>'
        '<label><input type="checkbox" id="t_anchovy"> Anchovy</label></div>'
        '<button id="submitBtn">Place Order</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{'
        '  const want=["t_mushroom","t_olive","t_pepper"];'
        '  const got=Array.from(document.querySelectorAll("input[type=checkbox]")).filter(c=>c.checked).map(c=>c.id);'
        '  const ok=got.length===want.length && want.every(w=>got.includes(w));'
        '  return ok?"toppings_correct":"incorrect";};'
    )

    # S-E16: Click a numbered button in a grid
    emit(
        "S_E16_button_grid.html",
        "Click Button 5",
        '<h1>Click button labelled "5"</h1>'
        '<div style="display:grid;grid-template-columns:repeat(3,80px);gap:8px;margin-top:16px">'
        + ''.join(f'<button id="b{i}" data-n="{i}">{i}</button>' for i in range(1, 10))
        + '</div>',
        'window.__clicked=null;'
        'document.querySelectorAll("button[data-n]").forEach(b=>b.onclick=()=>{window.__clicked=b.dataset.n;});'
        'window.getTaskResult=()=>window.__clicked==="5"?"button5_correct":"incorrect";'
    )

    # S-E17: Find email in a profile
    emit(
        "S_E17_find_email.html",
        "Find Email",
        '<h1>User Profile</h1>'
        '<div style="background:#f8fafc;padding:14px;border-radius:8px">'
        '<p>Username: <strong>arielnguyen</strong></p>'
        '<p>Display name: Ariel Nguyen</p>'
        '<p>Email: ariel.nguyen@northpine.io</p>'
        '<p>Joined: 2024-03-12</p></div>'
        '<label>Type the email address:</label>'
        '<input type="email" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v==="ariel.nguyen@northpine.io"?"email_correct":"incorrect";};'
    )

    # S-E18: Increment quantity to 4
    emit(
        "S_E18_increment_qty.html",
        "Set Quantity to 4",
        '<h1>Set quantity to 4 and click Add to Cart</h1>'
        '<div style="margin:14px 0">'
        '<button id="dec">−</button>'
        '<input type="number" id="qty" value="1" min="0" style="width:80px;text-align:center">'
        '<button id="inc">+</button></div>'
        '<button id="addBtn">Add to Cart</button>',
        'const q=document.getElementById("qty");'
        'document.getElementById("inc").onclick=()=>{q.value=parseInt(q.value||0)+1;};'
        'document.getElementById("dec").onclick=()=>{q.value=Math.max(0,parseInt(q.value||0)-1);};'
        'window.__added=false;'
        'document.getElementById("addBtn").onclick=()=>{window.__added=true;window.__qty=q.value;};'
        'window.getTaskResult=()=>{'
        ' if(!window.__added)return "not_added";'
        ' return window.__qty==="4"?"qty_correct":"wrong_qty_"+window.__qty;};'
    )

    # S-E19: Identify the cheapest item from a list
    emit(
        "S_E19_cheapest.html",
        "Cheapest",
        '<h1>Pricing Page</h1>'
        '<table><tr><th>Plan</th><th>Monthly</th></tr>'
        '<tr><td>Basic</td><td>$8.99</td></tr>'
        '<tr><td>Standard</td><td>$14.99</td></tr>'
        '<tr><td>Pro</td><td>$24.99</td></tr>'
        '<tr><td>Lite</td><td>$4.99</td></tr></table>'
        '<label>Name of the cheapest plan:</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v==="lite"?"cheapest_correct":"incorrect";};'
    )

    # S-E20: Match company to founding year
    emit(
        "S_E20_match_year.html",
        "Match Year",
        '<h1>Company Founding Years</h1>'
        '<ul><li>Northpine Inc. — 2009</li><li>Greypath LLC — 1997</li>'
        '<li>Brassroot Co. — 2015</li><li>Ironvale GmbH — 1982</li></ul>'
        '<label>Which year was Brassroot Co. founded?</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim();'
        ' return v==="2015"?"year_correct":"incorrect";};'
    )

    # S-E21: Reorder by clicking arrows
    emit(
        "S_E21_select_color.html",
        "Select Color",
        '<h1>Choose the color "teal"</h1>'
        '<select id="color"><option value="">—</option>'
        '<option value="red">red</option><option value="blue">blue</option>'
        '<option value="teal">teal</option><option value="green">green</option>'
        '<option value="purple">purple</option></select>'
        '<button id="submitBtn">Confirm</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("color").value;'
        ' return v==="teal"?"color_correct":"incorrect";};'
    )

    # S-E22: Find a date in a sentence
    emit(
        "S_E22_find_date.html",
        "Find Date",
        '<h1>Project Notice</h1>'
        '<p style="font-size:15px;line-height:1.7;background:#f8fafc;padding:14px;border-radius:8px">'
        'The hardware procurement window for Q3 closed on June 14th. '
        'All purchase requests submitted after that date will be deferred to Q4. '
        'Final invoices must be reconciled by August 30th to remain in this fiscal year.</p>'
        '<label>What is the procurement deadline date? (e.g. June 14)</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return ["june 14","june 14th","6/14","jun 14"].includes(v)?"date_correct":"incorrect";};'
    )


# ── MEDIUM tasks (16 new) ─────────────────────────────────────────────

def make_medium():
    # S-M1: Conditional fill (find row in table, then fill another field with that row's value)
    emit(
        "S_M1_conditional_fill.html",
        "Conditional Fill",
        '<h1>Find the rate for Standard shipping in zone B and enter it.</h1>'
        '<table><tr><th>Method</th><th>Zone A</th><th>Zone B</th><th>Zone C</th></tr>'
        '<tr><td>Express</td><td>$18</td><td>$22</td><td>$28</td></tr>'
        '<tr><td>Standard</td><td>$8</td><td>$11</td><td>$14</td></tr>'
        '<tr><td>Economy</td><td>$4</td><td>$6</td><td>$8</td></tr></table>'
        '<label>Rate in dollars (number only):</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().replace("$","");'
        ' return v==="11"?"rate_correct":"incorrect";};'
    )

    # S-M2: Multi-step form
    emit(
        "S_M2_address_form.html",
        "Address Form",
        '<h1>Shipping Address</h1>'
        '<p class="instructions">Fill in: 240 Maple Ave, Portland, OR 97205</p>'
        '<label>Street</label><input type="text" id="street">'
        '<label>City</label><input type="text" id="city">'
        '<label>State</label><select id="state"><option value="">—</option><option>CA</option><option>NY</option><option>OR</option><option>WA</option></select>'
        '<label>ZIP</label><input type="text" id="zip">'
        '<button id="submitBtn">Save Address</button>',
        'window.__saved=null;'
        'document.getElementById("submitBtn").onclick=()=>{window.__saved={s:document.getElementById("street").value,c:document.getElementById("city").value,st:document.getElementById("state").value,z:document.getElementById("zip").value};};'
        'window.getTaskResult=()=>{if(!window.__saved)return "not_saved";'
        ' const a=window.__saved; const ok=a.s.toLowerCase()==="240 maple ave"&&a.c.toLowerCase()==="portland"&&a.st==="OR"&&a.z==="97205";'
        ' return ok?"address_correct":"incorrect";};'
    )

    # S-M3: Calculate tax
    emit(
        "S_M3_calculate_tax.html",
        "Calculate Tax",
        '<h1>Calculate sales tax</h1>'
        '<p>Subtotal: $80.00. Tax rate: 7.5%. Compute the tax (in dollars).</p>'
        '<label>Tax amount (e.g. 6.00):</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().replace("$","");'
        ' const n=parseFloat(v); return Math.abs(n-6.00)<0.005?"tax_correct":"incorrect_"+v;};'
    )

    # S-M4: Sort by clicking column header (DOM remains static, only sort indicator)
    emit(
        "S_M4_filter_employees.html",
        "Filter Employees",
        '<h1>Find the engineer with the longest tenure</h1>'
        '<table><tr><th>Name</th><th>Role</th><th>Years</th></tr>'
        '<tr><td>Sara Patel</td><td>Engineer</td><td>4</td></tr>'
        '<tr><td>Mark Lin</td><td>Designer</td><td>9</td></tr>'
        '<tr><td>Ana Roth</td><td>Engineer</td><td>11</td></tr>'
        '<tr><td>Tom Vega</td><td>Engineer</td><td>2</td></tr>'
        '<tr><td>Eli Cole</td><td>Manager</td><td>14</td></tr></table>'
        '<label>Name:</label><input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v==="ana roth"?"employee_correct":"incorrect";};'
    )

    # S-M5: Password requirements
    emit(
        "S_M5_password.html",
        "Password Validator",
        '<h1>Choose a valid password</h1>'
        '<p>Password must satisfy: at least 8 characters, one uppercase, one digit, one symbol from <code>!@#$</code>.</p>'
        '<label>Password</label><input type="text" id="pw"><button id="submitBtn">Set Password</button>',
        'window.__set=null;'
        'document.getElementById("submitBtn").onclick=()=>{window.__set=document.getElementById("pw").value;};'
        'window.getTaskResult=()=>{if(window.__set==null)return "not_set"; const p=window.__set;'
        ' const ok=p.length>=8 && /[A-Z]/.test(p) && /\\d/.test(p) && /[!@#$]/.test(p);'
        ' return ok?"password_valid":"invalid";};'
    )

    # S-M6: Two factor select + confirm
    emit(
        "S_M6_two_factor.html",
        "Two-Factor",
        '<h1>Enable two-factor authentication via Authenticator app</h1>'
        '<div><label><input type="radio" name="m" value="sms"> SMS</label>'
        '<label><input type="radio" name="m" value="email"> Email</label>'
        '<label><input type="radio" name="m" value="app"> Authenticator app</label>'
        '<label><input type="radio" name="m" value="key"> Hardware key</label></div>'
        '<button id="enableBtn">Enable 2FA</button>'
        '<div id="confirm" style="display:none;margin-top:16px"><p>Confirm activation?</p><button id="confirmBtn">Confirm</button></div>',
        'window.__enabled=null;'
        'document.getElementById("enableBtn").onclick=()=>{const r=document.querySelector("input[name=m]:checked");'
        ' if(!r)return; document.getElementById("confirm").style.display="block"; window.__pending=r.value;};'
        'document.getElementById("confirmBtn").onclick=()=>{window.__enabled=window.__pending;};'
        'window.getTaskResult=()=>{if(!window.__enabled)return "not_confirmed";'
        ' return window.__enabled==="app"?"twofa_correct":"wrong_method_"+window.__enabled;};'
    )

    # S-M7: Read PDF-style invoice
    emit(
        "S_M7_invoice_total.html",
        "Invoice Total",
        '<h1>Invoice Subtotal Calculator</h1>'
        '<table><tr><th>Item</th><th>Qty</th><th>Unit</th></tr>'
        '<tr><td>Widget A</td><td>3</td><td>$8.00</td></tr>'
        '<tr><td>Widget B</td><td>2</td><td>$11.50</td></tr>'
        '<tr><td>Service Fee</td><td>1</td><td>$5.25</td></tr></table>'
        '<label>Total subtotal (sum of qty×unit):</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().replace("$","");'
        ' const n=parseFloat(v); return Math.abs(n-52.25)<0.005?"total_correct":"incorrect_"+v;};'
    )

    # S-M8: Toggle 3 settings
    emit(
        "S_M8_settings_toggle.html",
        "Toggle Settings",
        '<h1>Privacy Settings</h1>'
        '<p>Enable: <strong>Two-Step Login</strong> and <strong>Marketing Emails</strong>. Disable: <strong>Activity Sharing</strong>.</p>'
        '<label><input type="checkbox" id="s_login" checked> Two-Step Login (currently OFF→ON)</label>'
        '<label><input type="checkbox" id="s_share" checked> Activity Sharing (currently ON→OFF)</label>'
        '<label><input type="checkbox" id="s_market"> Marketing Emails (currently OFF→ON)</label>'
        '<button id="submitBtn">Save Settings</button>'
        '<script>document.getElementById("s_login").checked=false;'
        'document.getElementById("s_share").checked=true;'
        'document.getElementById("s_market").checked=false;</script>',
        'window.__saved=null;'
        'document.getElementById("submitBtn").onclick=()=>{window.__saved={l:document.getElementById("s_login").checked,s:document.getElementById("s_share").checked,m:document.getElementById("s_market").checked};};'
        'window.getTaskResult=()=>{if(!window.__saved)return "not_saved";'
        ' const a=window.__saved; const ok=a.l===true && a.s===false && a.m===true;'
        ' return ok?"settings_correct":"incorrect";};'
    )

    # S-M9: Schedule meeting form (date+time+attendee)
    emit(
        "S_M9_schedule_meeting.html",
        "Schedule Meeting",
        '<h1>Schedule a meeting on 2026-06-15 at 14:30 with attendee mariah@firm.io</h1>'
        '<label>Date</label><input type="text" id="date">'
        '<label>Time (HH:MM 24h)</label><input type="text" id="time">'
        '<label>Attendee email</label><input type="email" id="att">'
        '<button id="submitBtn">Schedule</button>',
        'window.__scheduled=null;'
        'document.getElementById("submitBtn").onclick=()=>{window.__scheduled={d:document.getElementById("date").value,t:document.getElementById("time").value,a:document.getElementById("att").value};};'
        'window.getTaskResult=()=>{if(!window.__scheduled)return "not_scheduled";'
        ' const a=window.__scheduled; const ok=a.d==="2026-06-15"&&a.t==="14:30"&&a.a.toLowerCase()==="mariah@firm.io";'
        ' return ok?"scheduled_correct":"incorrect";};'
    )

    # S-M10: Conditional reasoning over text
    emit(
        "S_M10_policy_reasoning.html",
        "Policy Reasoning",
        '<h1>Refund Policy</h1>'
        '<p style="background:#f8fafc;padding:14px;border-radius:8px">'
        'Refunds are issued in full within 14 days of purchase. Between 15 and 30 days, '
        'a 25% restocking fee applies. After 30 days, no refunds are issued.</p>'
        '<p>A customer purchased on day 0 and is requesting a refund on day 22. '
        'How many cents are deducted from a $200.00 purchase as the restocking fee?</p>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().replace("$","").replace(",","");'
        ' const n=parseFloat(v); return (Math.abs(n-5000)<1 || Math.abs(n-50)<0.005)?"fee_correct":"incorrect_"+v;};'
    )

    # S-M11: Find a code in a long text
    emit(
        "S_M11_find_code.html",
        "Find Reference Code",
        '<h1>Email transcript</h1>'
        '<div style="background:#f8fafc;padding:14px;border-radius:8px;font-family:monospace;font-size:13px;line-height:1.6">'
        'From: support@northpine.io<br>To: customer@example.com<br>'
        'Subject: Your replacement order<br><br>'
        'Hi — your replacement has shipped. Reference number is RX-7B4K-29CT. '
        'Please mention this in any further correspondence. Tracking will follow separately.'
        '</div>'
        '<label>Reference number:</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toUpperCase();'
        ' return v==="RX-7B4K-29CT"?"code_correct":"incorrect";};'
    )

    # S-M12: Multi-rule eligibility
    emit(
        "S_M12_eligibility.html",
        "Eligibility Check",
        '<h1>Loan Eligibility</h1>'
        '<p>Eligibility: applicant must be ≥21 years old, have an income ≥$45,000, '
        'and a credit score ≥680.</p>'
        '<p>Applicant: 27 years old, $52,000 income, credit score 663. Eligible? (yes/no)</p>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v==="no"?"eligibility_correct":"incorrect";};'
    )

    # S-M13: Convert units
    emit(
        "S_M13_unit_conversion.html",
        "Unit Conversion",
        '<h1>Convert 7.5 kilometers to meters</h1>'
        '<input type="text" id="ans" placeholder="number only"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().replace(",","");'
        ' const n=parseFloat(v); return Math.abs(n-7500)<1?"conversion_correct":"incorrect_"+v;};'
    )

    # S-M14: Pick winner from a leaderboard
    emit(
        "S_M14_leaderboard.html",
        "Leaderboard Winner",
        '<h1>Tournament Leaderboard</h1>'
        '<table><tr><th>Player</th><th>Score</th></tr>'
        '<tr><td>player_red</td><td>1240</td></tr>'
        '<tr><td>player_blue</td><td>1880</td></tr>'
        '<tr><td>player_green</td><td>1700</td></tr>'
        '<tr><td>player_yellow</td><td>2050</td></tr>'
        '<tr><td>player_violet</td><td>1995</td></tr></table>'
        '<label>Winner username:</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v==="player_yellow"?"winner_correct":"incorrect";};'
    )

    # S-M15: Pick all items above threshold
    emit(
        "S_M15_above_threshold.html",
        "Above Threshold",
        '<h1>Tag all stocks priced above $100</h1>'
        '<p>Check the box next to every stock with closing price &gt; $100.</p>'
        '<label><input type="checkbox" id="aapl"> AAPL — $182.50</label>'
        '<label><input type="checkbox" id="ge"> GE — $74.10</label>'
        '<label><input type="checkbox" id="msft"> MSFT — $422.30</label>'
        '<label><input type="checkbox" id="t"> T — $18.95</label>'
        '<label><input type="checkbox" id="amd"> AMD — $124.40</label>'
        '<button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{'
        ' const want=["aapl","msft","amd"]; '
        ' const got=Array.from(document.querySelectorAll("input[type=checkbox]")).filter(c=>c.checked).map(c=>c.id);'
        ' const ok=got.length===want.length && want.every(w=>got.includes(w));'
        ' return ok?"threshold_correct":"incorrect";};'
    )

    # S-M16: Choose nearest match (text similarity)
    emit(
        "S_M16_choose_match.html",
        "Choose Match",
        '<h1>Match the user query to the closest tag</h1>'
        '<p>Query: <em>"how do I revoke API access for an old service account?"</em></p>'
        '<label><input type="radio" name="tag" value="billing"> billing</label>'
        '<label><input type="radio" name="tag" value="api-credentials" id="r_api"> api-credentials</label>'
        '<label><input type="radio" name="tag" value="onboarding"> onboarding</label>'
        '<label><input type="radio" name="tag" value="logs"> logs</label>'
        '<button id="submitBtn">Submit</button>',
        'window.__chosen=null;'
        'document.getElementById("submitBtn").onclick=()=>{const r=document.querySelector("input[name=tag]:checked"); window.__chosen=r?r.value:null;};'
        'window.getTaskResult=()=>window.__chosen==="api-credentials"?"match_correct":"incorrect";'
    )


# ── HARD tasks (12 new) ────────────────────────────────────────────────

def make_hard():
    # S-H1: Three-line resolution to compute final
    emit(
        "S_H1_invoice_workflow.html",
        "Invoice Workflow",
        '<h1>Compute the final invoice amount</h1>'
        '<p>Subtotal of line items: $480.00. Apply a 10% volume discount; then apply 6% tax to the discounted amount; then add a flat $12 shipping fee. Round to two decimal places. Enter only the final number.</p>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{'
        ' const v=document.getElementById("ans").value.trim().replace("$","").replace(",","");'
        ' const expected=480*0.9*1.06+12; const n=parseFloat(v);'
        ' return Math.abs(n-expected)<0.02?"final_correct":"incorrect_"+v;};'
    )

    # S-H2: Plan dispatch with constraints
    emit(
        "S_H2_dispatch.html",
        "Dispatch Decision",
        '<h1>Choose the route</h1>'
        '<p>Three routes are available:</p>'
        '<table><tr><th>Route</th><th>Distance (km)</th><th>Tolls ($)</th><th>Notes</th></tr>'
        '<tr><td>R-Alpha</td><td>340</td><td>22</td><td>No restrictions</td></tr>'
        '<tr><td>R-Bravo</td><td>290</td><td>40</td><td>Closed on weekends</td></tr>'
        '<tr><td>R-Charlie</td><td>355</td><td>0</td><td>Hazmat-only</td></tr></table>'
        '<p>Today is Saturday. Cargo is non-hazardous. Choose the shortest legal route.</p>'
        '<input type="text" id="ans" placeholder="route name e.g. R-Alpha"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toUpperCase();'
        ' return (v==="R-ALPHA")?"route_correct":"incorrect_"+v;};'
    )

    # S-H3: Cipher decode
    emit(
        "S_H3_decode_message.html",
        "Decode Message",
        '<h1>Decode the message</h1>'
        '<p>The message has been ROT-13 encoded. Decode it and type the plaintext.</p>'
        '<p style="font-family:monospace;background:#f8fafc;padding:12px;border-radius:6px">URYYB JBEYQ</p>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toUpperCase();'
        ' return v==="HELLO WORLD"?"decode_correct":"incorrect_"+v;};'
    )

    # S-H4: Choose subset to maximize within budget (knapsack-lite)
    emit(
        "S_H4_budget_select.html",
        "Budget Selection",
        '<h1>Pick items totaling exactly $50</h1>'
        '<p>Select any combination of these items so the total equals exactly $50.</p>'
        '<label><input type="checkbox" id="i_a" data-p="35"> A — $35</label>'
        '<label><input type="checkbox" id="i_b" data-p="20"> B — $20</label>'
        '<label><input type="checkbox" id="i_c" data-p="15"> C — $15</label>'
        '<label><input type="checkbox" id="i_d" data-p="10"> D — $10</label>'
        '<label><input type="checkbox" id="i_e" data-p="5"> E — $5</label>'
        '<button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{'
        ' let total=0; document.querySelectorAll("input[type=checkbox]").forEach(c=>{if(c.checked)total+=parseInt(c.dataset.p)});'
        ' return total===50?"budget_correct":"incorrect_total_"+total;};'
    )

    # S-H5: Compose multi-field email reply
    emit(
        "S_H5_compose_reply.html",
        "Compose Reply",
        '<h1>Reply to email</h1>'
        '<div style="background:#f8fafc;padding:12px;border-radius:6px;font-size:14px">'
        'From: alex.romero@example.com<br>'
        'To: support@northpine.io<br>'
        'Subject: missing license key for invoice INV-9821<br><br>'
        'Hi, I purchased a Pro license under invoice INV-9821 yesterday but never received the activation key. Could you resend it? — Alex'
        '</div>'
        '<label>To:</label><input type="email" id="to">'
        '<label>Subject:</label><input type="text" id="subj">'
        '<label>Reply body (must mention the invoice number INV-9821):</label>'
        '<textarea id="body" rows="3"></textarea>'
        '<button id="submitBtn">Send Reply</button>',
        'window.__sent=null;'
        'document.getElementById("submitBtn").onclick=()=>{window.__sent={to:document.getElementById("to").value,subj:document.getElementById("subj").value,body:document.getElementById("body").value};};'
        'window.getTaskResult=()=>{if(!window.__sent)return "not_sent";'
        ' const m=window.__sent;'
        ' const ok=m.to.toLowerCase()==="alex.romero@example.com" && /re:?/i.test(m.subj) && m.body.includes("INV-9821");'
        ' return ok?"reply_correct":"incorrect";};'
    )

    # S-H6: Sequence puzzle
    emit(
        "S_H6_sequence.html",
        "Sequence",
        '<h1>What number comes next?</h1>'
        '<p style="font-size:24px;font-weight:600">2, 6, 12, 20, 30, ?</p>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim();'
        ' return v==="42"?"sequence_correct":"incorrect_"+v;};'
    )

    # S-H7: Multi-criteria sort
    emit(
        "S_H7_sort_priority.html",
        "Sort by Priority",
        '<h1>Pick the issue to triage first</h1>'
        '<p>Triage rule: critical severity wins; if tied, the older ticket wins.</p>'
        '<table><tr><th>ID</th><th>Severity</th><th>Opened</th></tr>'
        '<tr><td>T-001</td><td>high</td><td>2026-01-04</td></tr>'
        '<tr><td>T-002</td><td>critical</td><td>2026-04-22</td></tr>'
        '<tr><td>T-003</td><td>medium</td><td>2025-12-30</td></tr>'
        '<tr><td>T-004</td><td>critical</td><td>2026-02-11</td></tr>'
        '<tr><td>T-005</td><td>high</td><td>2025-11-09</td></tr></table>'
        '<input type="text" id="ans" placeholder="ticket id e.g. T-001"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toUpperCase();'
        ' return v==="T-004"?"triage_correct":"incorrect_"+v;};'
    )

    # S-H8: Currency conversion chain
    emit(
        "S_H8_currency.html",
        "Currency Chain",
        '<h1>Currency conversion</h1>'
        '<p>Rates: 1 USD = 0.92 EUR. 1 EUR = 142 JPY. Convert $250 USD into JPY (round to nearest yen).</p>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().replace(",","").replace("¥","");'
        ' const n=parseInt(v); const expected=Math.round(250*0.92*142);'
        ' return Math.abs(n-expected)<=1?"currency_correct":"incorrect_"+v;};'
    )

    # S-H9: Schema mapping
    emit(
        "S_H9_schema_map.html",
        "Schema Mapping",
        '<h1>Map source fields to target field names</h1>'
        '<p>Source record: <code>{firstName, lastName, telephone, dob}</code></p>'
        '<p>Target record uses: <code>given_name, family_name, phone, birthdate</code></p>'
        '<label>Which target field maps from <code>dob</code>?</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v==="birthdate"?"map_correct":"incorrect_"+v;};'
    )

    # S-H10: Compute median
    emit(
        "S_H10_median.html",
        "Median",
        '<h1>Compute the median</h1>'
        '<p>Numbers: 4, 11, 9, 22, 7, 18, 6</p>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim();'
        ' return v==="9"?"median_correct":"incorrect_"+v;};'
    )

    # S-H11: Path planning text
    emit(
        "S_H11_path.html",
        "Shortest Path",
        '<h1>Find the shortest path A→D in this directed graph</h1>'
        '<pre style="font-family:monospace;background:#f8fafc;padding:14px;border-radius:6px">'
        'A->B (4)\nA->C (2)\nC->B (1)\nB->D (3)\nC->D (10)</pre>'
        '<label>Total distance:</label><input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim();'
        ' return v==="6"?"path_correct":"incorrect_"+v;};'
    )

    # S-H12: Audit log filter
    emit(
        "S_H12_audit.html",
        "Audit Log",
        '<h1>Audit log</h1>'
        '<pre style="font-family:monospace;background:#f8fafc;padding:14px;border-radius:6px;font-size:13px">'
        '2026-04-30 09:12  user=alice  action=login         ip=10.0.1.5\n'
        '2026-04-30 09:14  user=bob    action=upload_file   ip=10.0.1.7\n'
        '2026-04-30 09:18  user=alice  action=delete_file   ip=10.0.1.5\n'
        '2026-04-30 09:21  user=carol  action=login         ip=10.0.2.4\n'
        '2026-04-30 09:24  user=bob    action=delete_file   ip=10.0.1.7\n'
        '2026-04-30 09:31  user=alice  action=download_file ip=10.0.1.5</pre>'
        '<label>How many distinct users performed a delete action?</label>'
        '<input type="text" id="ans"><button id="submitBtn">Submit</button>',
        'document.getElementById("submitBtn").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim();'
        ' return v==="2"?"audit_correct":"incorrect_"+v;};'
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_easy()
    make_medium()
    make_hard()
    print("Wrote:", len(list(OUT_DIR.glob("S_E1[1-9]*.html") )) + len(list(OUT_DIR.glob("S_E2[0-2]*.html"))) + len(list(OUT_DIR.glob("S_M*.html"))) + len(list(OUT_DIR.glob("S_H*.html"))), "static tasks")

if __name__ == "__main__":
    main()
