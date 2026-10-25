"""
Synthetic bilingual sentiment data generator for Unilyf.
Languages: en (English), hi (Hindi, Devanagari), hinglish (Hindi in Roman script)
Labels: 0 = negative, 1 = neutral, 2 = positive
Usage: python generate_unilyf_data.py --per_cell 600 --out dataset/unilyf_sentiment.csv
"""
import argparse, csv, random

NEG, NEU, POS = 0, 1, 2

# BASE[topic][label][lang] -> list of seed sentences
BASE = {
 "exams": {
  NEG: {
   "en": ["Exam schedule got changed again with just one day notice, this is so frustrating.",
          "The result is still not out and nobody from the office is replying.",
          "Question paper was way out of syllabus, everyone is upset."],
   "hi": ["परीक्षा की तारीख फिर से बदल दी गई, सिर्फ एक दिन पहले बताया, बहुत गुस्सा आ रहा है।",
          "रिज़ल्ट अभी तक नहीं आया और ऑफिस में कोई जवाब नहीं दे रहा।",
          "पेपर सिलेबस से बाहर था, सब बहुत परेशान हैं।"],
   "hinglish": ["Exam ki date phir se change ho gayi, sirf ek din pehle bataya, bahut frustrating hai.",
                "Result abhi tak nahi aaya aur office mein koi reply hi nahi kar raha.",
                "Paper syllabus se bahar tha yaar, sab bahut upset hain."]},
  NEU: {
   "en": ["Mid-sem exams will begin from next Monday as per the notice.",
          "The date sheet has been uploaded on the portal.",
          "Admit cards will be available from the exam cell tomorrow."],
   "hi": ["मिड-सेम परीक्षा अगले सोमवार से शुरू होगी।",
          "डेट शीट पोर्टल पर अपलोड कर दी गई है।",
          "एडमिट कार्ड कल परीक्षा विभाग से मिलेंगे।"],
   "hinglish": ["Mid-sem exams agle Monday se start honge, notice lag gaya hai.",
                "Date sheet portal pe upload ho gayi hai.",
                "Admit card kal exam cell se milenge."]},
  POS: {
   "en": ["Great news, the results are out and the portal worked smoothly!",
          "Exam went really well and the invigilators were very helpful.",
          "Happy that the revaluation got cleared quickly."],
   "hi": ["बहुत अच्छी खबर, रिज़ल्ट आ गया और पोर्टल भी बढ़िया चला!",
          "परीक्षा बहुत अच्छी रही और निरीक्षक भी बहुत मददगार थे।",
          "खुशी है कि रीवैल्यूएशन जल्दी हो गया।"],
   "hinglish": ["Mast news, result aa gaya aur portal bhi smooth chala!",
                "Exam bahut accha gaya aur invigilators bhi kaafi helpful the.",
                "Khushi hai ki revaluation jaldi clear ho gaya."]},
 },
 "hostel": {
  NEG: {
   "en": ["Hostel water supply has been cut since morning and no one is fixing it.",
          "The mess food quality is getting worse every week, really disappointing."],
   "hi": ["हॉस्टल में सुबह से पानी बंद है और कोई ठीक नहीं कर रहा।",
          "मेस के खाने की क्वालिटी हर हफ्ते खराब होती जा रही है, बहुत निराशाजनक है।"],
   "hinglish": ["Hostel mein subah se paani band hai aur koi theek nahi kar raha.",
                "Mess ke khane ki quality har hafte kharab hoti ja rahi hai, bahut disappointing."]},
  NEU: {
   "en": ["Hostel room allotment list will be displayed on the notice board.",
          "Hostel curfew timing is 10 PM for first-year students."],
   "hi": ["हॉस्टल रूम आवंटन की सूची नोटिस बोर्ड पर लगाई जाएगी।",
          "पहले वर्ष के छात्रों के लिए हॉस्टल का समय रात 10 बजे तक है।"],
   "hinglish": ["Hostel room allotment ki list notice board pe lagayi jayegi.",
                "First-year students ke liye hostel curfew raat 10 baje ka hai."]},
  POS: {
   "en": ["The new hostel rooms are clean and spacious, loving it!",
          "Warden fixed my room issue within an hour, great support."],
   "hi": ["नए हॉस्टल के कमरे साफ और बड़े हैं, बहुत पसंद आए!",
          "वार्डन ने मेरी समस्या एक घंटे में ठीक कर दी, बहुत बढ़िया सहयोग।"],
   "hinglish": ["Naye hostel ke rooms saaf aur bade hain, bahut pasand aaye!",
                "Warden ne meri problem ek ghante mein solve kar di, kya baat hai."]},
 },
 "canteen": {
  NEG: {
   "en": ["Canteen prices went up again and the portions are tiny.",
          "Waited 40 minutes in the canteen queue and the food was cold."],
   "hi": ["कैंटीन में दाम फिर बढ़ गए और खाना बहुत कम मिलता है।",
          "कैंटीन की लाइन में 40 मिनट खड़े रहे और खाना ठंडा था।"],
   "hinglish": ["Canteen ke rates phir badh gaye aur portion bahut chhota hai.",
                "Canteen ki line mein 40 minute wait kiya aur khana thanda tha."]},
  NEU: {
   "en": ["Canteen will remain closed on Sunday for maintenance.",
          "A new snack counter has been opened near the library."],
   "hi": ["रखरखाव के लिए कैंटीन रविवार को बंद रहेगी।",
          "लाइब्रेरी के पास एक नया स्नैक काउंटर खोला गया है।"],
   "hinglish": ["Maintenance ke liye canteen Sunday ko band rahegi.",
                "Library ke paas ek naya snack counter khula hai."]},
  POS: {
   "en": ["The new canteen menu is amazing, the paneer wrap is so good!",
          "Super quick service at the canteen today, thank you team."],
   "hi": ["कैंटीन का नया मेन्यू शानदार है, पनीर रैप बहुत स्वादिष्ट है!",
          "आज कैंटीन में बहुत जल्दी सर्विस मिली, धन्यवाद टीम।"],
   "hinglish": ["Canteen ka naya menu ekdum zabardast hai, paneer wrap bahut tasty hai!",
                "Aaj canteen mein super fast service mili, thanks team."]},
 },
 "wifi": {
  NEG: {
   "en": ["Campus WiFi is down again during online assignment submission, unacceptable.",
          "Internet speed in the labs is painfully slow."],
   "hi": ["ऑनलाइन असाइनमेंट जमा करते समय कैंपस वाईफाई फिर बंद हो गया, यह ठीक नहीं है।",
          "लैब में इंटरनेट की स्पीड बहुत ही धीमी है।"],
   "hinglish": ["Online assignment submission ke time campus WiFi phir down ho gaya, bilkul theek nahi hai.",
                "Labs mein internet speed bahut hi slow hai."]},
  NEU: {
   "en": ["WiFi maintenance is scheduled tonight from 12 to 2 AM.",
          "Students can get new WiFi passwords from the IT desk."],
   "hi": ["आज रात 12 से 2 बजे तक वाईफाई का रखरखाव होगा।",
          "छात्र नया वाईफाई पासवर्ड आईटी डेस्क से ले सकते हैं।"],
   "hinglish": ["Aaj raat 12 se 2 baje tak WiFi maintenance hoga.",
                "Students naya WiFi password IT desk se le sakte hain."]},
  POS: {
   "en": ["WiFi speed has improved a lot after the upgrade, finally!",
          "Smooth internet in the library today, got so much work done."],
   "hi": ["अपग्रेड के बाद वाईफाई की स्पीड बहुत बढ़िया हो गई है, आखिरकार!",
          "आज लाइब्रेरी में इंटरनेट बहुत अच्छा चला, खूब काम हुआ।"],
   "hinglish": ["Upgrade ke baad WiFi speed bahut badhiya ho gayi hai, finally!",
                "Aaj library mein internet ekdum smooth tha, bahut kaam ho gaya."]},
 },
 "events": {
  NEG: {
   "en": ["The fest was badly organised, no proper seating and the sound kept failing.",
          "Event got cancelled last minute and nobody informed us."],
   "hi": ["फेस्ट का आयोजन बहुत खराब था, बैठने की जगह नहीं थी और साउंड बार-बार बंद हुआ।",
          "इवेंट आखिरी समय पर रद्द कर दिया गया और किसी ने बताया नहीं।"],
   "hinglish": ["Fest ka organisation bahut kharab tha, seating nahi thi aur sound baar-baar fail hua.",
                "Event last minute pe cancel ho gaya aur kisi ne bataya bhi nahi."]},
  NEU: {
   "en": ["The cultural fest registrations open from Monday.",
          "Tech talk on AI will be held in the main auditorium at 3 PM."],
   "hi": ["सांस्कृतिक फेस्ट के रजिस्ट्रेशन सोमवार से शुरू होंगे।",
          "एआई पर टेक टॉक दोपहर 3 बजे मुख्य ऑडिटोरियम में होगी।"],
   "hinglish": ["Cultural fest ke registrations Monday se open honge.",
                "AI pe tech talk main auditorium mein 3 baje hogi."]},
  POS: {
   "en": ["The annual fest was incredible, best night of the semester!",
          "Loved the hackathon, great mentors and amazing teams."],
   "hi": ["सालाना फेस्ट शानदार रहा, इस सेमेस्टर की सबसे अच्छी रात!",
          "हैकाथॉन बहुत पसंद आया, मेंटर और टीमें बहुत बढ़िया थीं।"],
   "hinglish": ["Annual fest kamaal ka tha, semester ki sabse best raat!",
                "Hackathon bahut pasand aaya, mentors aur teams ekdum mast the."]},
 },
 "placements": {
  NEG: {
   "en": ["Placement cell is not sharing any updates and we are all anxious.",
          "The company cancelled the drive after we prepared for weeks, so disappointing."],
   "hi": ["प्लेसमेंट सेल कोई अपडेट नहीं दे रहा और हम सब चिंतित हैं।",
          "हफ्तों की तैयारी के बाद कंपनी ने ड्राइव रद्द कर दी, बहुत निराशा हुई।"],
   "hinglish": ["Placement cell koi update share nahi kar raha aur hum sab tension mein hain.",
                "Hafton ki preparation ke baad company ne drive cancel kar di, bahut bura laga."]},
  NEU: {
   "en": ["A new company will visit the campus next week for placements.",
          "Resume submission deadline is Friday evening."],
   "hi": ["अगले हफ्ते एक नई कंपनी प्लेसमेंट के लिए कैंपस आएगी।",
          "रिज़्यूमे जमा करने की अंतिम तारीख शुक्रवार शाम है।"],
   "hinglish": ["Agle hafte ek nayi company placement ke liye campus aayegi.",
                "Resume submission ki deadline Friday shaam tak hai."]},
  POS: {
   "en": ["Got placed! The placement team really supported us throughout.",
          "Highest package this year, so proud of our batch!"],
   "hi": ["प्लेसमेंट हो गया! प्लेसमेंट टीम ने हर कदम पर साथ दिया।",
          "इस साल का सबसे बड़ा पैकेज, अपने बैच पर बहुत गर्व है!"],
   "hinglish": ["Placement ho gaya! Placement team ne har step pe bahut support kiya.",
                "Is saal ka highest package, apne batch pe bahut proud hoon!"]},
 },
 "fees": {
  NEG: {
   "en": ["Fee portal keeps crashing and I got charged a late fine for no fault of mine.",
          "Sudden fee hike announced without any explanation, students are angry."],
   "hi": ["फीस पोर्टल बार-बार क्रैश हो रहा है और मेरी गलती के बिना लेट फाइन लग गया।",
          "बिना किसी स्पष्टीकरण के अचानक फीस बढ़ा दी गई, छात्र नाराज़ हैं।"],
   "hinglish": ["Fee portal baar-baar crash ho raha hai aur meri galti ke bina late fine laga diya.",
                "Bina kisi explanation ke achanak fees badha di gayi, students naraz hain."]},
  NEU: {
   "en": ["Last date for fee payment is 30th November.",
          "Scholarship forms are available at the accounts office."],
   "hi": ["फीस जमा करने की अंतिम तारीख 30 नवंबर है।",
          "स्कॉलरशिप फॉर्म अकाउंट्स ऑफिस में उपलब्ध हैं।"],
   "hinglish": ["Fee payment ki last date 30 November hai.",
                "Scholarship forms accounts office mein available hain."]},
  POS: {
   "en": ["Fee payment was so easy this time, done in two minutes!",
          "Thankful for the scholarship, it really helps my family."],
   "hi": ["इस बार फीस भरना बहुत आसान था, दो मिनट में हो गया!",
          "स्कॉलरशिप के लिए आभारी हूँ, इससे मेरे परिवार को बहुत मदद मिली।"],
   "hinglish": ["Is baar fee payment bahut easy tha, do minute mein ho gaya!",
                "Scholarship ke liye thankful hoon, mere ghar walon ko bahut help mili."]},
 },
 "library": {
  NEG: {
   "en": ["Library AC is not working and it is impossible to study in this heat.",
          "Important reference books are always unavailable, so annoying."],
   "hi": ["लाइब्रेरी का एसी खराब है और इस गर्मी में पढ़ना असंभव है।",
          "ज़रूरी रेफरेंस किताबें हमेशा उपलब्ध नहीं होतीं, बहुत परेशान करने वाला है।"],
   "hinglish": ["Library ka AC kharab hai aur is garmi mein padhna impossible hai.",
                "Zaroori reference books hamesha available nahi hoti, bahut annoying hai."]},
  NEU: {
   "en": ["Library will remain open till midnight during exams.",
          "New books have been added to the computer science section."],
   "hi": ["परीक्षा के दौरान लाइब्रेरी आधी रात तक खुली रहेगी।",
          "कंप्यूटर साइंस सेक्शन में नई किताबें जोड़ी गई हैं।"],
   "hinglish": ["Exams ke dauran library raat 12 baje tak khuli rahegi.",
                "Computer science section mein nayi books add ki gayi hain."]},
  POS: {
   "en": ["The new reading room is so peaceful, perfect for studying!",
          "Librarian helped me find exactly what I needed, super helpful."],
   "hi": ["नया रीडिंग रूम बहुत शांत है, पढ़ाई के लिए एकदम सही!",
          "लाइब्रेरियन ने मुझे ठीक वही किताब ढूँढने में मदद की, बहुत मददगार।"],
   "hinglish": ["Naya reading room bahut peaceful hai, padhai ke liye ekdum perfect!",
                "Librarian ne mujhe exactly wahi book dhoondhne mein help ki, bahut helpful."]},
 },
}

PREFIX = {
 "en": ["", "", "", "Update: ", "Heads up: ", "Just now: ", "Honestly, ", "FYI "],
 "hi": ["", "", "", "अपडेट: ", "ध्यान दें: ", "अभी-अभी: ", "सच कहूँ तो "],
 "hinglish": ["", "", "", "Update: ", "Heads up: ", "Abhi abhi: ", "Sach bolun toh ", "Bhai "],
}
SUFFIX = {
 "en": ["", "", "", " #campuslife", " #unilyf", " Anyone else facing this?", " Please share."],
 "hi": ["", "", "", " #कैंपस", " #unilyf", " क्या और लोग भी यही झेल रहे हैं?", " कृपया शेयर करें।"],
 "hinglish": ["", "", "", " #campuslife", " #unilyf", " Aur kisi ko bhi hua?", " Please share karo."],
}
EMOJI = {
 NEG: ["", "", "😡", "😞", "😤", "😔", "👎"],
 NEU: ["", "", "", "📢", "📌", "🗓️", "ℹ️"],
 POS: ["", "", "😍", "🎉", "👏", "😊", "🔥"],
}
TYPO_SWAPS = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("bahut", "bohot"), ("hai", "h"),
              ("nahi", "nhi"), ("kya", "kia"), ("please", "pls"), ("thanks", "thnx")]


def noisy(text, lang, rng):
    if lang == "hinglish":
        if rng.random() < 0.35:
            text = text.lower()
        if rng.random() < 0.3:
            a, b = rng.choice(TYPO_SWAPS)
            text = text.replace(a, b, 1)
    elif lang == "en":
        if rng.random() < 0.15:
            text = text.lower()
        if rng.random() < 0.1:
            text = text.replace("please", "pls").replace("Please", "Pls")
    return text


def build(seed_sentence, lang, label, rng):
    t = rng.choice(PREFIX[lang]) + seed_sentence + rng.choice(SUFFIX[lang])
    t = noisy(t, lang, rng)
    e = rng.choice(EMOJI[label])
    if e:
        t = t + " " + e
    return " ".join(t.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cell", type=int, default=600,
                    help="target rows per (language, label) cell")
    ap.add_argument("--out", default="dataset/unilyf_sentiment.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows, seen = [], set()
    for lang in ("en", "hi", "hinglish"):
        for label in (NEG, NEU, POS):
            pool = [(topic, s) for topic, d in BASE.items() for s in d[label][lang]]
            made, tries = 0, 0
            while made < a.per_cell and tries < a.per_cell * 30:
                tries += 1
                topic, s = rng.choice(pool)
                text = build(s, lang, label, rng)
                if text in seen:
                    continue
                seen.add(text)
                rows.append({"text": text, "label": label, "language": lang, "topic": topic})
                made += 1

    rng.shuffle(rows)
    n = len(rows)
    for i, r in enumerate(rows):
        r["id"] = i
        r["split"] = "train" if i < 0.8 * n else ("val" if i < 0.9 * n else "test")

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label", "language", "topic", "split"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {n} rows to {a.out}")


if __name__ == "__main__":
    main()
