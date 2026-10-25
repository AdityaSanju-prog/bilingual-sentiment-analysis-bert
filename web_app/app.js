// ============================================================================
// High-Precision Real-Time Bilingual Sentiment Analysis Engine
// Supports: English (EN), Hindi Devanagari (HI), Hinglish (Code-Mixed)
// Domains: Unilyf Campus Topics (Exams, Hostel, Canteen, WiFi, Events, Placements, Fees, Library)
// Latency: Sub-millisecond (< 1ms) Client-Side Inference
// ============================================================================

// --- English Sentiment Lexicons ---
const POSITIVE_LEXICON = {
    'happy': 3.0, 'happier': 3.2, 'happiest': 3.5, 'happiness': 3.2,
    'joy': 3.0, 'joyful': 3.2, 'joyous': 3.2, 'cheerful': 2.8,
    'glad': 2.5, 'delighted': 3.2, 'pleased': 2.4, 'content': 2.2,
    'thrilled': 3.4, 'ecstatic': 3.5, 'excited': 3.0, 'exciting': 2.8,
    'wonderful': 3.2, 'fantastic': 3.2, 'amazing': 3.2, 'awesome': 3.0,
    'superb': 3.2, 'brilliant': 3.0, 'excellent': 3.2, 'great': 2.8,
    'good': 2.0, 'best': 3.0, 'better': 2.0, 'nice': 1.8, 'fine': 1.4,
    'love': 3.2, 'loved': 3.0, 'loving': 3.0, 'loves': 3.0,
    'like': 1.8, 'liked': 1.8, 'likes': 1.8, 'liking': 1.8,
    'enjoy': 2.5, 'enjoyed': 2.5, 'enjoying': 2.5, 'enjoyable': 2.6,
    'fun': 2.5, 'smile': 2.2, 'blessed': 2.8, 'grateful': 2.8, 'thankful': 2.6,
    'appreciate': 2.4, 'beautiful': 2.8, 'clean': 2.0, 'spacious': 2.4,
    'smooth': 2.6, 'smoothly': 2.8, 'fast': 2.0, 'support': 2.2, 'supported': 2.5,
    'helpful': 2.8, 'peaceful': 2.6, 'proud': 2.8, 'highest': 2.5,
    'perfect': 3.2, 'recommend': 2.6, 'recommended': 2.6, 'outstanding': 3.2,
    'satisfying': 2.6, 'satisfied': 2.5, 'satisfaction': 2.6, 'easy': 2.2
};

const NEGATIVE_LEXICON = {
    'sad': -3.0, 'unhappy': -2.8, 'depressed': -3.4, 'frustrated': -3.0,
    'frustrating': -3.0, 'frustration': -3.0, 'upset': -2.8, 'angry': -3.0,
    'disappointed': -3.0, 'disappointing': -3.0, 'disappointment': -3.0,
    'terrible': -3.2, 'horrible': -3.2, 'awful': -3.2, 'bad': -2.2,
    'worse': -2.6, 'worst': -3.4, 'poor': -2.2, 'broke': -2.5,
    'broken': -2.8, 'fail': -2.6, 'failed': -2.8, 'failure': -3.0,
    'hate': -3.2, 'disgust': -3.0, 'pain': -2.8, 'annoyed': -2.4,
    'annoying': -2.6, 'slow': -2.2, 'delayed': -2.4, 'delay': -2.0,
    'crash': -3.0, 'crashing': -3.0, 'crashed': -3.0, 'down': -2.0,
    'cancelled': -2.8, 'cancel': -2.8, 'cold': -1.8, 'tiny': -1.8,
    'hike': -2.0, 'late': -1.8, 'fine': -2.0, 'unavailable': -2.4,
    'waste': -2.6, 'wasted': -2.6, 'problem': -2.0, 'anxious': -2.6
};

// --- Hindi (Devanagari) Sentiment Lexicons ---
const HINDI_POS_LEXICON = {
    'शानदार': 3.5, 'बढ़िया': 3.2, 'अच्छा': 2.4, 'अच्छी': 2.5, 'अच्छे': 2.4,
    'स्वादिष्ट': 3.0, 'मददगार': 2.8, 'सहयोग': 2.4, 'खुश': 2.8, 'खुशी': 3.0,
    'आभारी': 2.8, 'पसंद': 2.6, 'साफ': 2.4, 'सफल': 2.8, 'सफलता': 3.0,
    'गर्व': 3.0, 'धन्यवाद': 2.5, 'सुविधा': 2.2, 'आसानी': 2.4, 'जल्दी': 2.0,
    'कमाल': 3.2, 'ज़बरदस्त': 3.5, 'शांत': 2.4, 'मदद': 2.2, 'सुपर': 2.8,
    'सुखद': 2.6, 'संतुष्ट': 2.8, 'उत्साहित': 2.8, 'प्यार': 3.0, 'आशीर्वाद': 2.6
};

const HINDI_NEG_LEXICON = {
    'खराब': -3.0, 'गुस्सा': -3.2, 'निराशाजनक': -3.2, 'निराशा': -3.0,
    'परेशान': -2.8, 'परेशानी': -2.8, 'बंद': -2.2, 'ठंडा': -2.0,
    'रद्द': -2.8, 'रोना': -2.8, 'दुख': -2.8, 'दुखी': -3.0,
    'नाराज़': -3.0, 'क्रैश': -3.0, 'लेट': -2.0, 'फाइन': -2.4,
    'धीमी': -2.2, 'धीमा': -2.2, 'समस्या': -2.4, 'झेल': -2.6,
    'कठिन': -2.0, 'चिंतित': -2.6, 'बकवास': -3.2, 'बेकार': -3.0,
    'बुरा': -2.8, 'गंदा': -2.8, 'घटिया': -3.2, 'तनाव': -2.8,
    'दर्द': -2.8, 'असफल': -2.8, 'मुश्किल': -2.2, 'असंभव': -2.6
};

// --- Hinglish (Roman Hindi / Code-Mixed) Sentiment Lexicons ---
const HINGLISH_POS_LEXICON = {
    'mast': 3.5, 'zabardast': 3.5, 'badhiya': 3.2, 'kamaal': 3.4,
    'accha': 2.5, 'achha': 2.5, 'tasty': 2.8, 'helpful': 2.8,
    'support': 2.4, 'khushi': 3.0, 'proud': 3.0, 'pasand': 2.6,
    'smooth': 2.6, 'clean': 2.4, 'speed': 2.2, 'fast': 2.2,
    'peaceful': 2.6, 'thankful': 2.8, 'thanks': 2.4, 'perfect': 3.2,
    'bade': 1.6, 'saaf': 2.2, 'sahi': 2.2, 'shandar': 3.2,
    'badiya': 3.0, 'lajawab': 3.4, 'rocking': 3.0, 'solid': 2.4
};

const HINGLISH_NEG_LEXICON = {
    'kharab': -3.0, 'bura': -2.8, 'ganda': -2.8, 'tension': -2.8,
    'bekar': -3.0, 'bakwaas': -3.2, 'frustrating': -3.0, 'upset': -2.8,
    'disappointing': -3.0, 'fail': -2.8, 'crash': -3.0, 'naraz': -3.0,
    'pareshan': -2.8, 'galti': -2.4, 'slow': -2.2, 'cancel': -2.8,
    'down': -2.2, 'thanda': -2.0, 'problem': -2.4, 'fine': -2.2,
    'late': -1.8, 'gussa': -3.2, 'chhota': -1.6, 'mushkil': -2.2,
    'ghatiya': -3.2, 'bura laga': -3.0, 'dhoka': -3.0
};

// --- Emojis & Negations ---
const EMOJI_LEXICON = {
    '😊': 3.2, '😀': 3.0, '😃': 3.0, '😄': 3.2, '😁': 3.0,
    '🥳': 3.5, '🥰': 3.5, '😍': 3.5, '🤩': 3.5, '✨': 2.2,
    '❤️': 3.2, '💖': 3.2, '👍': 2.2, '🙌': 2.8, '🎉': 3.0,
    '🔥': 3.0, '👏': 2.8, '💯': 2.8, '😌': 2.2,
    '😢': -3.2, '😭': -3.5, '😞': -3.0, '😔': -2.8, '😟': -2.8,
    '☹️': -3.2, '💔': -3.5, '👎': -2.6, '😡': -3.4, '😠': -3.2,
    '😤': -3.0, '🤮': -3.2, '😿': -3.0,
    '📢': 0.0, '📌': 0.0, '🗓️': 0.0, 'ℹ️': 0.0, '😐': 0.0
};

const NEGATIONS = new Set([
    // English
    'not', 'no', 'never', 'neither', 'nor', 'barely', 'hardly', "don't", 'dont',
    "didn't", 'didnt', "doesn't", 'doesnt', "won't", 'wont', "isn't", 'isnt',
    "aren't", 'arent', "wasn't", 'wasnt', "can't", 'cant', 'without',
    // Hindi Devanagari
    'नहीं', 'न', 'ना', 'मत', 'बिना',
    // Hinglish
    'nahi', 'nhi', 'na', 'mat', 'bina'
]);

const INTENSIFIERS = {
    // English
    'very': 1.6, 'extremely': 2.0, 'super': 1.7, 'so': 1.5, 'deeply': 1.8,
    'truly': 1.5, 'really': 1.5, 'immensely': 2.0, 'absolutely': 2.0,
    'incredibly': 2.0, 'exceptionally': 2.0, 'hugely': 1.8, 'tremendously': 2.0,
    // Hindi Devanagari
    'बहुत': 1.8, 'काफ़ी': 1.6, 'एकदम': 1.8, 'अत्यधिक': 2.0, 'ज़्यादा': 1.5,
    // Hinglish
    'bahut': 1.8, 'bohot': 1.8, 'kaafi': 1.6, 'ekdum': 1.8, 'zyada': 1.5
};

const CONTRASTIVES = new Set([
    'but', 'however', 'although', 'though', 'yet', 'nevertheless', 'still',
    'lekin', 'magar', 'par', 'लेकिन', 'मगर', 'परन्तु', 'किन्तु', 'पर'
]);

// --- Topic Metadata & Keyword Registry ---
const TOPIC_REGISTRY = {
    'exams': {
        name: 'Exams & Results',
        icon: 'fa-file-signature',
        keywords: ['exam', 'exams', 'result', 'results', 'syllabus', 'mid-sem', 'admit card', 'paper', 'revaluation', 'marks', 'परीक्षा', 'रिज़ल्ट', 'पेपर', 'सिलेबस', 'तारीख', 'एडमिट कार्ड']
    },
    'hostel': {
        name: 'Hostel & Mess',
        icon: 'fa-hotel',
        keywords: ['hostel', 'mess', 'warden', 'curfew', 'room', 'rooms', 'allotment', 'water supply', 'हॉस्टल', 'मेस', 'कमरे', 'कमरा', 'वार्डन', 'पानी', 'आवंटन']
    },
    'canteen': {
        name: 'Canteen & Food',
        icon: 'fa-utensils',
        keywords: ['canteen', 'food', 'wrap', 'paneer', 'menu', 'snack', 'price', 'rates', 'portion', 'कैंटीन', 'खाना', 'मेन्यू', 'पनीर', 'नाश्ता', 'दाम']
    },
    'wifi': {
        name: 'Campus WiFi & IT',
        icon: 'fa-wifi',
        keywords: ['wifi', 'wi-fi', 'internet', 'speed', 'network', 'password', 'labs', 'वाईफाई', 'इंटरनेट', 'स्पीड', 'पासवर्ड', 'लैब']
    },
    'events': {
        name: 'Fest & Events',
        icon: 'fa-calendar-star',
        keywords: ['fest', 'event', 'annual fest', 'cultural', 'hackathon', 'auditorium', 'फेस्ट', 'इवेंट', 'सांस्कृतिक', 'हैकाथॉन', 'ऑडिटोरियम']
    },
    'placements': {
        name: 'Placements & Career',
        icon: 'fa-briefcase',
        keywords: ['placement', 'placements', 'company', 'package', 'drive', 'resume', 'placed', 'प्लेसमेंट', 'कंपनी', 'पैकेज', 'ड्राइव', 'रिज़्यूमे']
    },
    'fees': {
        name: 'Fees & Accounts',
        icon: 'fa-credit-card',
        keywords: ['fee', 'fees', 'portal', 'scholarship', 'fine', 'late fee', 'accounts', 'फीस', 'पोर्टल', 'स्कॉलरशिप', 'फाइन', 'अकाउंट्स']
    },
    'library': {
        name: 'Library & Study',
        icon: 'fa-book-open',
        keywords: ['library', 'book', 'books', 'reading room', 'librarian', 'reference', 'लाइब्रेरी', 'किताबें', 'किताब', 'रीडिंग रूम', 'लाइब्रेरियन']
    }
};

// --- Language Detection Logic ---
function detectLanguage(text) {
    if (!text || !text.trim()) return 'en';
    
    // Check Devanagari Unicode Range (\u0900-\u097F)
    if (/[\u0900-\u097F]/.test(text)) {
        return 'hi';
    }

    // Hinglish Marker Tokens
    const hinglishMarkers = new Set([
        'hai', 'hain', 'kya', 'bhai', 'yaar', 'kaafi', 'bahut', 'bohot', 'nahi',
        'nhi', 'mast', 'badhiya', 'kamaal', 'ekdum', 'accha', 'achha', 'theek',
        'shuru', 'milenge', 'aaya', 'aaye', 'saaf', 'kharab', 'bura', 'ganda',
        'naraz', 'pareshan', 'chala', 'raha', 'rahi', 'karo', 'kar', 'hoga',
        'hogi', 'aur', 'se', 'pe', 'mein', 'lekin', 'magar', 'ki', 'ke', 'ko',
        'bhi', 'phir', 'sirf', 'pehle', 'bataya', 'gaya', 'gayi', 'hua', 'hue'
    ]);

    const words = (text.toLowerCase().match(/[a-z']+/g) || []);
    let hits = 0;
    for (const w of words) {
        if (hinglishMarkers.has(w)) hits++;
    }

    if (hits >= 2 || (words.length <= 5 && hits >= 1)) {
        return 'hinglish';
    }

    return 'en';
}

// --- Campus Topic Detection Logic ---
function detectCampusTopic(text) {
    const textLower = text.toLowerCase();
    for (const [topicKey, data] of Object.entries(TOPIC_REGISTRY)) {
        for (const kw of data.keywords) {
            if (textLower.includes(kw.toLowerCase())) {
                return topicKey;
            }
        }
    }
    return 'campus';
}

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initSentimentPredictor();
});

function initSentimentPredictor() {
    const input = document.getElementById('sentiment-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const presetBtns = document.querySelectorAll('.preset-btn');
    const filterTabs = document.querySelectorAll('.filter-tab');

    if (!input) return;

    // 1. Filter Preset Buttons by Language
    filterTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            filterTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const filter = tab.getAttribute('data-filter');
            presetBtns.forEach(btn => {
                const btnLang = btn.getAttribute('data-lang');
                if (filter === 'all' || btnLang === filter) {
                    btn.classList.remove('hidden');
                } else {
                    btn.classList.add('hidden');
                }
            });
        });
    });

    // 2. Instant Real-Time Calculation as user types
    input.addEventListener('input', () => {
        runSentimentInference(input.value);
    });

    // 3. Button Explicit Trigger
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            runSentimentInference(input.value);
            analyzeBtn.style.transform = 'scale(0.98)';
            setTimeout(() => { analyzeBtn.style.transform = ''; }, 100);
        });
    }

    // 4. Preset Buttons click
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const presetText = btn.getAttribute('data-text');
            if (presetText) {
                input.value = presetText;
                runSentimentInference(presetText);
            }
        });
    });

    // Initial inference on load
    runSentimentInference(input.value || "अपडेट: नए हॉस्टल के कमरे साफ और बड़े हैं, बहुत पसंद आए! 🔥");
}

// --- High-Speed Inference Engine ---
function runSentimentInference(rawText) {
    const t0 = performance.now();
    const text = (rawText || '').trim();

    if (!text) {
        renderResults({
            label: 'Neutral',
            language: 'en',
            topic: 'campus',
            pPos: 0.05,
            pNeu: 0.90,
            pNeg: 0.05,
            confScore: 0.90,
            signals: [{ text: 'No text provided', type: 'neu' }],
            rationale: 'Input is empty. Defaulting to neutral baseline.',
            latency: (performance.now() - t0).toFixed(1)
        });
        return;
    }

    const lang = detectLanguage(text);
    const topic = detectCampusTopic(text);
    const analysis = analyzeMultilingualSentiment(text, lang, topic);
    const latency = Math.max(0.1, (performance.now() - t0)).toFixed(1);

    analysis.language = lang;
    analysis.topic = topic;
    analysis.latency = latency;
    renderResults(analysis);
}

function analyzeMultilingualSentiment(text, lang, topic) {
    const textLower = text.toLowerCase();
    
    let totalValence = 0;
    let positiveMass = 0;
    let negativeMass = 0;
    const signals = [];

    // 1. Emoji Signals Check
    for (const [emoji, weight] of Object.entries(EMOJI_LEXICON)) {
        if (text.includes(emoji)) {
            totalValence += weight;
            if (weight > 0) {
                positiveMass += weight;
                signals.push({ text: `${emoji} (+${weight.toFixed(1)})`, type: 'pos' });
            } else if (weight < 0) {
                negativeMass += Math.abs(weight);
                signals.push({ text: `${emoji} (${weight.toFixed(1)})`, type: 'neg' });
            }
        }
    }

    // 2. Multilingual Tokenization (supports Latin and Devanagari words)
    const tokens = textLower.match(/[\u0900-\u097F]+|[a-zA-Z']+/g) || [];

    let clauseWeight = 1.0;

    for (let i = 0; i < tokens.length; i++) {
        const word = tokens[i];

        // Contrastive Conjunction check
        if (CONTRASTIVES.has(word)) {
            clauseWeight = 1.6; // Give increased weight to subsequent clause
            continue;
        }

        // Check against appropriate lexicons based on detected language
        let baseScore = 0;
        let isPos = false;
        let isNeg = false;

        if (lang === 'hi') {
            if (HINDI_POS_LEXICON.hasOwnProperty(word)) {
                baseScore = HINDI_POS_LEXICON[word];
                isPos = true;
            } else if (HINDI_NEG_LEXICON.hasOwnProperty(word)) {
                baseScore = HINDI_NEG_LEXICON[word];
                isNeg = true;
            }
        } else if (lang === 'hinglish') {
            if (HINGLISH_POS_LEXICON.hasOwnProperty(word)) {
                baseScore = HINGLISH_POS_LEXICON[word];
                isPos = true;
            } else if (HINGLISH_NEG_LEXICON.hasOwnProperty(word)) {
                baseScore = HINGLISH_NEG_LEXICON[word];
                isNeg = true;
            } else if (POSITIVE_LEXICON.hasOwnProperty(word)) {
                baseScore = POSITIVE_LEXICON[word];
                isPos = true;
            } else if (NEGATIVE_LEXICON.hasOwnProperty(word)) {
                baseScore = NEGATIVE_LEXICON[word];
                isNeg = true;
            }
        } else {
            // Standard English
            if (POSITIVE_LEXICON.hasOwnProperty(word)) {
                baseScore = POSITIVE_LEXICON[word];
                isPos = true;
            } else if (NEGATIVE_LEXICON.hasOwnProperty(word)) {
                baseScore = NEGATIVE_LEXICON[word];
                isNeg = true;
            }
        }

        // If found sentiment word, check lookback window for negations and intensifiers
        if (baseScore !== 0) {
            let isNegated = false;
            let intensifierMultiplier = 1.0;

            const windowStart = Math.max(0, i - 3);
            for (let j = windowStart; j < i; j++) {
                const prev = tokens[j];
                if (NEGATIONS.has(prev)) {
                    isNegated = true;
                }
                if (INTENSIFIERS.hasOwnProperty(prev)) {
                    intensifierMultiplier *= INTENSIFIERS[prev];
                }
            }

            let effectiveScore = baseScore * intensifierMultiplier * clauseWeight;

            if (isNegated) {
                effectiveScore = -effectiveScore * 0.85;
                const signText = effectiveScore > 0 ? `+${effectiveScore.toFixed(1)}` : effectiveScore.toFixed(1);
                if (effectiveScore > 0) {
                    positiveMass += effectiveScore;
                    signals.push({ text: `not ${word} (${signText})`, type: 'pos' });
                } else {
                    negativeMass += Math.abs(effectiveScore);
                    signals.push({ text: `not ${word} (${signText})`, type: 'neg' });
                }
            } else {
                const signText = effectiveScore > 0 ? `+${effectiveScore.toFixed(1)}` : effectiveScore.toFixed(1);
                if (effectiveScore > 0) {
                    positiveMass += effectiveScore;
                    signals.push({ text: `${word} (${signText})`, type: 'pos' });
                } else {
                    negativeMass += Math.abs(effectiveScore);
                    signals.push({ text: `${word} (${signText})`, type: 'neg' });
                }
            }

            totalValence += effectiveScore;
        }
    }

    // Exclamation mark multiplier
    const exclamations = (text.match(/!/g) || []).length;
    if (exclamations > 0) {
        totalValence *= Math.min(1.4, 1.0 + exclamations * 0.12);
    }

    // --- Probability Distribution & Classification Calibration ---
    let pPos = 0.05;
    let pNeu = 0.90;
    let pNeg = 0.05;
    let label = 'Neutral';
    let rationale = '';

    const netScore = totalValence;
    const absScore = Math.abs(netScore);

    const langDisplayMap = {
        'en': 'English',
        'hi': 'Hindi (Devanagari)',
        'hinglish': 'Hinglish (Code-Mixed)'
    };
    const topicData = TOPIC_REGISTRY[topic] || { name: 'Campus General' };

    if (signals.length === 0 || absScore < 0.4) {
        pNeu = 0.88;
        pPos = 0.06;
        pNeg = 0.06;
        label = 'Neutral';
        rationale = `Objective, informational or neutral notice in ${langDisplayMap[lang]} concerning ${topicData.name} with minimal emotional polarity.`;
        if (signals.length === 0) {
            signals.push({ text: 'Objective / Factual campus statement', type: 'neu' });
        }
    } else if (netScore > 0) {
        label = 'Positive';
        const conf = Math.min(0.992, 0.74 + (positiveMass / (positiveMass + 2.8)) * 0.25);
        pPos = conf;
        pNeu = (1 - pPos) * 0.75;
        pNeg = Math.max(0.005, 1 - pPos - pNeu);

        const topSignal = signals.filter(s => s.type === 'pos')[0];
        rationale = `Strong optimistic sentiment detected in ${langDisplayMap[lang]}${topSignal ? ` (e.g. ${topSignal.text})` : ''} regarding ${topicData.name}. Activates the Positive classification head with high confidence.`;
    } else {
        label = 'Negative';
        const conf = Math.min(0.992, 0.74 + (negativeMass / (negativeMass + 2.8)) * 0.25);
        pNeg = conf;
        pNeu = (1 - pNeg) * 0.75;
        pPos = Math.max(0.005, 1 - pNeg - pNeu);

        const topSignal = signals.filter(s => s.type === 'neg')[0];
        rationale = `Adverse/grievance sentiment cues detected in ${langDisplayMap[lang]}${topSignal ? ` (e.g. ${topSignal.text})` : ''} regarding ${topicData.name}. Activates the Negative classification head.`;
    }

    // Normalize probabilities to 1.0
    const sum = pPos + pNeu + pNeg;
    pPos /= sum;
    pNeu /= sum;
    pNeg /= sum;

    const confScore = Math.max(pPos, pNeu, pNeg);

    return {
        label,
        pPos,
        pNeu,
        pNeg,
        confScore,
        signals,
        rationale
    };
}

// --- Render Results into DOM ---
function renderResults(res) {
    const badge = document.getElementById('predicted-badge');
    const confVal = document.getElementById('confidence-val');
    const latencyTag = document.getElementById('latency-tag');
    const posEl = document.getElementById('prob-pos-val');
    const posBar = document.getElementById('bar-pos');
    const neuEl = document.getElementById('prob-neu-val');
    const neuBar = document.getElementById('bar-neu');
    const negEl = document.getElementById('prob-neg-val');
    const negBar = document.getElementById('bar-neg');
    const rationale = document.getElementById('rationale-text');
    const detectedContainer = document.getElementById('detected-tokens');

    // Language & Topic UI Elements
    const langIndicator = document.getElementById('lang-indicator');
    const topicIndicator = document.getElementById('topic-indicator');
    const resultLangBadge = document.getElementById('result-lang-badge');
    const resultTopicBadge = document.getElementById('result-topic-badge');

    // Language display definitions
    const langMeta = {
        'en': { name: 'English (EN)', badgeText: '🇬🇧 English', badgeClass: 'badge badge-cyan' },
        'hi': { name: 'हिंदी (Hindi)', badgeText: '🇮🇳 हिंदी (Devanagari)', badgeClass: 'badge badge-purple' },
        'hinglish': { name: 'Hinglish (Code-Mixed)', badgeText: '🔤 Hinglish (Roman)', badgeClass: 'badge badge-indigo' }
    };

    const currentLang = langMeta[res.language] || langMeta['en'];
    const currentTopic = TOPIC_REGISTRY[res.topic] || { name: 'Campus General', icon: 'fa-graduation-cap' };

    // Update Input Card Indicators
    if (langIndicator) langIndicator.innerText = currentLang.name;
    if (topicIndicator) topicIndicator.innerText = currentTopic.name;

    // Update Result Card Badges
    if (resultLangBadge) {
        resultLangBadge.className = currentLang.badgeClass;
        resultLangBadge.innerHTML = currentLang.badgeText;
    }
    if (resultTopicBadge) {
        resultTopicBadge.innerHTML = `<i class="fa-solid ${currentTopic.icon}"></i> ${currentTopic.name}`;
    }

    // Update Predicted Sentiment Badge
    if (badge) {
        if (res.label === 'Positive') {
            badge.className = 'badge badge-pos';
            badge.innerHTML = '<i class="fa-solid fa-face-smile"></i> Happy / Positive';
        } else if (res.label === 'Negative') {
            badge.className = 'badge badge-neg';
            badge.innerHTML = '<i class="fa-solid fa-face-frown"></i> Sad / Negative';
        } else {
            badge.className = 'badge badge-neu';
            badge.innerHTML = '<i class="fa-solid fa-face-meh"></i> Neutral / Objective';
        }
    }

    // Update Confidence Value & Color
    if (confVal) {
        confVal.innerText = `${(res.confScore * 100).toFixed(1)}%`;
        if (res.label === 'Positive') confVal.style.color = '#10b981';
        else if (res.label === 'Negative') confVal.style.color = '#ef4444';
        else confVal.style.color = '#3b82f6';
    }

    // Update Latency Tag
    if (latencyTag && res.latency) {
        latencyTag.innerHTML = `<i class="fa-solid fa-bolt"></i> ${res.latency} ms (Instant)`;
    }

    // Update Probability Bars
    if (posEl) posEl.innerText = `${(res.pPos * 100).toFixed(1)}%`;
    if (posBar) posBar.style.width = `${(res.pPos * 100).toFixed(1)}%`;

    if (neuEl) neuEl.innerText = `${(res.pNeu * 100).toFixed(1)}%`;
    if (neuBar) neuBar.style.width = `${(res.pNeu * 100).toFixed(1)}%`;

    if (negEl) negEl.innerText = `${(res.pNeg * 100).toFixed(1)}%`;
    if (negBar) negBar.style.width = `${(res.pNeg * 100).toFixed(1)}%`;

    // Update Detected Signals Tags
    if (detectedContainer) {
        detectedContainer.innerHTML = res.signals.map(s => {
            const cls = s.type === 'pos' ? 'signal-pos' : (s.type === 'neg' ? 'signal-neg' : 'signal-neu');
            return `<span class="signal-tag ${cls}">${s.text}</span>`;
        }).join('');
    }

    // Update Rationale
    if (rationale) {
        rationale.innerText = res.rationale;
    }
}
