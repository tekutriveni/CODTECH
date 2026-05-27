"""
Multi-Label Text Categorization using Machine Learning
=======================================================
A single news article can belong to MULTIPLE categories at once.
Example: "India wins cricket World Cup" → [Sports, India, Entertainment]

Pipeline:
  1. Dataset Creation  - 200 sample news headlines with multi-labels
  2. Preprocessing     - Cleaning, TF-IDF Vectorization
  3. Models            - OneVsRest with Logistic Regression, SVM, Random Forest
  4. Evaluation        - Hamming Loss, F1 (micro/macro), Accuracy
  5. Prediction        - Predict labels for new custom text
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (hamming_loss, f1_score,
                             accuracy_score, classification_report)
import json
import re
import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════
# STEP 1 — DATASET
# 200 headlines, each tagged with 1–3 labels from 8 categories
# ════════════════════════════════════════════════════════════

CATEGORIES = ['Politics', 'Sports', 'Technology', 'Business',
              'Health', 'Entertainment', 'Science', 'India']

RAW_DATA = [
    # Politics
    ("Parliament passes new education bill amid protests", ['Politics', 'India']),
    ("Prime Minister announces infrastructure development plan", ['Politics', 'India', 'Business']),
    ("Election commission sets voting date for state elections", ['Politics', 'India']),
    ("Opposition parties unite against government policy", ['Politics', 'India']),
    ("President signs landmark climate change agreement", ['Politics', 'Science']),
    ("Government introduces new tax reform bill in parliament", ['Politics', 'Business']),
    ("Diplomatic talks between nations resolve border tension", ['Politics']),
    ("New cabinet reshuffle announced by ruling party", ['Politics', 'India']),
    ("Supreme court delivers verdict on electoral bonds case", ['Politics', 'India']),
    ("Foreign minister visits neighboring country for peace talks", ['Politics']),
    ("State government launches welfare scheme for farmers", ['Politics', 'India']),
    ("Anti-corruption drive leads to arrests of senior officials", ['Politics', 'India']),
    ("Parliament budget session begins with heated debate", ['Politics', 'India']),
    ("Prime minister attends G20 summit in South Africa", ['Politics', 'India']),
    ("Local elections see record voter turnout across cities", ['Politics', 'India']),

    # Sports
    ("India wins cricket World Cup final against Australia", ['Sports', 'India']),
    ("Virat Kohli scores century in test match against England", ['Sports', 'India']),
    ("Neeraj Chopra wins gold medal at World Athletics Championship", ['Sports', 'India']),
    ("IPL season kicks off with Chennai Super Kings vs Mumbai Indians", ['Sports', 'India', 'Entertainment']),
    ("Serena Williams announces retirement from professional tennis", ['Sports']),
    ("FIFA World Cup 2026 draw announced by football federation", ['Sports']),
    ("PV Sindhu wins bronze medal at All England Badminton", ['Sports', 'India']),
    ("Formula One driver breaks lap record at Monaco Grand Prix", ['Sports']),
    ("India defeats Pakistan in Asia Cup cricket tournament", ['Sports', 'India']),
    ("Ronaldo scores hat-trick as Portugal wins European championship", ['Sports']),
    ("Olympic games officially open in Paris with grand ceremony", ['Sports', 'Entertainment']),
    ("Sunil Chhetri retires from Indian national football team", ['Sports', 'India']),
    ("Chess prodigy from India becomes youngest world champion", ['Sports', 'India']),
    ("Hockey India wins gold at Commonwealth Games", ['Sports', 'India']),
    ("Basketball star signs record-breaking contract with NBA team", ['Sports', 'Business']),

    # Technology
    ("Apple launches new iPhone with revolutionary AI features", ['Technology', 'Business']),
    ("Google unveils next generation Gemini AI language model", ['Technology', 'Business']),
    ("Tesla announces fully autonomous self-driving car software", ['Technology', 'Business']),
    ("Microsoft integrates ChatGPT into Office productivity suite", ['Technology', 'Business']),
    ("Meta releases open source large language model for developers", ['Technology']),
    ("India launches 5G network across 100 cities simultaneously", ['Technology', 'India', 'Business']),
    ("Cybersecurity breach exposes data of millions of users", ['Technology']),
    ("New quantum computer achieves breakthrough computational speed", ['Technology', 'Science']),
    ("Amazon introduces drone delivery service in urban areas", ['Technology', 'Business']),
    ("OpenAI announces GPT-5 with improved reasoning capabilities", ['Technology']),
    ("Startup builds AI tool to detect fake news automatically", ['Technology', 'India']),
    ("Government bans Chinese tech apps citing security concerns", ['Technology', 'Politics', 'India']),
    ("Reliance Jio launches affordable smartphone for rural India", ['Technology', 'India', 'Business']),
    ("Neural interface allows paralyzed patient to control computer", ['Technology', 'Health', 'Science']),
    ("Deepfake detection software launched by cybersecurity firm", ['Technology']),

    # Business
    ("Sensex hits all-time high crossing 80000 points milestone", ['Business', 'India']),
    ("Adani Group announces major expansion into renewable energy", ['Business', 'India']),
    ("RBI cuts interest rates to boost economic growth", ['Business', 'India', 'Politics']),
    ("Tata Motors reports record quarterly profit", ['Business', 'India']),
    ("Infosys wins multi-billion dollar IT contract from US firm", ['Business', 'India']),
    ("Stock market crashes after global recession fears emerge", ['Business']),
    ("Amazon acquires Indian ecommerce startup for two billion dollars", ['Business', 'Technology', 'India']),
    ("Gold prices reach record high amid global uncertainty", ['Business']),
    ("Zomato expands food delivery to 500 new cities in India", ['Business', 'India', 'Technology']),
    ("Startup ecosystem in Bangalore raises record venture capital", ['Business', 'India', 'Technology']),
    ("India GDP growth rate surpasses China for second year", ['Business', 'India']),
    ("Petrol prices hiked by five rupees after crude oil rise", ['Business', 'India']),
    ("SEBI introduces new rules for IPO listing norms", ['Business', 'India', 'Politics']),
    ("Flipkart announces massive sale with discounts across categories", ['Business', 'India']),
    ("Foreign direct investment in India reaches record high", ['Business', 'India']),

    # Health
    ("New vaccine developed for deadly dengue fever strain", ['Health', 'Science']),
    ("WHO declares end of COVID-19 as global health emergency", ['Health']),
    ("AIIMS researchers discover breakthrough cancer treatment drug", ['Health', 'Science', 'India']),
    ("Mental health awareness campaign launched by government", ['Health', 'India', 'Politics']),
    ("Yoga and meditation reduce stress by 40 percent study finds", ['Health', 'India', 'Science']),
    ("New diabetes drug shows promising results in clinical trials", ['Health', 'Science']),
    ("Air pollution linked to rise in respiratory diseases in cities", ['Health', 'India']),
    ("Government bans junk food advertisements targeting children", ['Health', 'Politics', 'India']),
    ("Blood pressure drug recalled due to contamination concerns", ['Health']),
    ("Exercise for 30 minutes daily reduces heart disease risk", ['Health', 'Science']),
    ("India achieves polio-free status for consecutive fifth year", ['Health', 'India']),
    ("New hospital with AI diagnostics opens in rural village", ['Health', 'Technology', 'India']),
    ("Study reveals link between social media use and depression", ['Health', 'Technology', 'Science']),
    ("Organ transplant success rate improves with new technique", ['Health', 'Science']),
    ("Government launches free health insurance for poor families", ['Health', 'Politics', 'India']),

    # Entertainment
    ("Bollywood film breaks box office record on opening weekend", ['Entertainment', 'India']),
    ("AR Rahman wins Grammy for best original movie soundtrack", ['Entertainment', 'India']),
    ("Netflix India original series becomes global trending show", ['Entertainment', 'India', 'Technology']),
    ("Shah Rukh Khan announces new film with international director", ['Entertainment', 'India']),
    ("Music streaming platform reaches 100 million users in India", ['Entertainment', 'India', 'Technology', 'Business']),
    ("Hollywood actor wins Oscar for best performance in drama", ['Entertainment']),
    ("Stand-up comedian from India performs sold-out show in London", ['Entertainment', 'India']),
    ("New season of popular web series drops on OTT platform", ['Entertainment', 'Technology']),
    ("Indian classical dancer wins prestigious international award", ['Entertainment', 'India']),
    ("Video game tournament prize pool exceeds 10 million dollars", ['Entertainment', 'Technology', 'Business']),
    ("Cannes film festival selects Indian director's film for competition", ['Entertainment', 'India']),
    ("Reality TV show breaks TRP records for fourth week running", ['Entertainment', 'India']),
    ("Singer releases surprise album breaking streaming records", ['Entertainment']),
    ("Animated movie from Indian studio wins international award", ['Entertainment', 'India', 'Technology']),
    ("Theatre festival celebrates 50 years of Indian performing arts", ['Entertainment', 'India']),

    # Science
    ("ISRO successfully launches satellite to study climate change", ['Science', 'India', 'Technology']),
    ("Scientists discover new species of dinosaur in Argentina", ['Science']),
    ("NASA astronauts complete successful moon landing mission", ['Science', 'Technology']),
    ("Indian scientist wins Nobel Prize for physics research", ['Science', 'India']),
    ("New study reveals ocean acidification threatens coral reefs", ['Science']),
    ("Chandrayaan-4 mission to collect moon soil samples launched", ['Science', 'India', 'Technology']),
    ("Researchers develop biodegradable plastic from sugarcane", ['Science', 'India']),
    ("James Webb telescope captures image of most distant galaxy", ['Science', 'Technology']),
    ("Breakthrough fusion energy reactor achieves net positive output", ['Science', 'Technology']),
    ("Study shows microplastics found in human blood for first time", ['Health', 'Science']),
    ("Scientists clone endangered snow leopard to save species", ['Science']),
    ("Solar storm causes widespread disruption to GPS systems", ['Science', 'Technology']),
    ("New material conducts electricity with zero resistance at room temperature", ['Science', 'Technology']),
    ("Research links deforestation to increase in zoonotic diseases", ['Science', 'Health']),
    ("DRDO develops hypersonic missile with advanced navigation system", ['Science', 'India', 'Technology']),

    # Extra mixed
    ("IIT graduate wins startup competition with AI health app", ['Technology', 'Health', 'India', 'Business']),
    ("Cricket stadium gets smart technology upgrade for fans", ['Sports', 'Technology', 'India']),
    ("Actor turns politician wins state election by landslide", ['Entertainment', 'Politics', 'India']),
    ("Farmer protests disrupt highway traffic in northern states", ['Politics', 'India', 'Business']),
    ("Mobile gaming industry in India grows 30 percent annually", ['Technology', 'Business', 'India', 'Entertainment']),
    ("Doctors use robotic surgery to perform complex heart operation", ['Health', 'Technology', 'Science']),
    ("Climate change causes severe drought in southern India", ['Science', 'India', 'Politics']),
    ("E-sports team from India qualifies for world championship", ['Sports', 'India', 'Technology', 'Entertainment']),
    ("Renewable energy company raises 500 crore in funding round", ['Business', 'India', 'Science']),
    ("Language learning app gains 10 million users in India", ['Technology', 'India', 'Business']),
    ("Study reveals benefits of traditional Indian diet for longevity", ['Health', 'India', 'Science']),
    ("Government mandates electric vehicles for public transport", ['Politics', 'India', 'Technology', 'Business']),
    ("Film industry raises funds for flood relief in coastal states", ['Entertainment', 'India']),
    ("Data privacy law passed to protect citizens online", ['Politics', 'Technology', 'India']),
    ("Bio-technology startup develops low-cost water purifier", ['Science', 'Technology', 'India', 'Business']),
    ("National sports academy produces five Olympic medal winners", ['Sports', 'India']),
    ("Social media influencer fined for misleading health claims", ['Health', 'Technology', 'India', 'Politics']),
    ("Indian railway network gets AI-powered predictive maintenance", ['Technology', 'India', 'Business']),
    ("University launches free online coding course for rural youth", ['Technology', 'India', 'Business']),
    ("Dengue cases spike in major cities during monsoon season", ['Health', 'India']),

    # More headlines to reach 200
    ("Prime Minister inaugurates new metro line in capital city", ['Politics', 'India', 'Technology']),
    ("World's largest solar farm commissioned in Rajasthan desert", ['Science', 'India', 'Business']),
    ("National cricket team coach announces new training strategy", ['Sports', 'India']),
    ("Smartphone addiction among teenagers raises mental health concerns", ['Health', 'Technology']),
    ("Budget airline launches cheap flights connecting tier 2 cities", ['Business', 'India']),
    ("Mars rover discovers evidence of ancient water beneath surface", ['Science', 'Technology']),
    ("Online fraud cases rise 50 percent according to cybercrime report", ['Technology', 'India']),
    ("Government school gets smart classroom technology upgrade", ['Technology', 'India', 'Politics']),
    ("Documentary on Indian wildlife wins international Emmy Award", ['Entertainment', 'India', 'Science']),
    ("Health ministry issues advisory on rising antibiotic resistance", ['Health', 'India', 'Politics']),
    ("Young swimmer from village breaks national record", ['Sports', 'India']),
    ("Automobile giant launches electric SUV with 600 km range", ['Technology', 'Business']),
    ("Scientists find cure for rare genetic disease using gene editing", ['Science', 'Health']),
    ("Social media platform banned in country over hate speech", ['Technology', 'Politics']),
    ("Tribal artist from Odisha gets Padma Shri for folk painting", ['Entertainment', 'India', 'Politics']),
    ("New highway connects remote villages reducing travel time", ['India', 'Politics', 'Business']),
    ("Bank introduces zero-fee digital payment for small merchants", ['Business', 'India', 'Technology']),
    ("Scientists measure strongest earthquake recorded in decade", ['Science']),
    ("Comedy film becomes surprise blockbuster earning 200 crores", ['Entertainment', 'India', 'Business']),
    ("Startup creates app to connect farmers directly with buyers", ['Business', 'India', 'Technology']),
    ("Student from rural India wins international math olympiad", ['India', 'Science']),
    ("Central bank launches digital rupee pilot across cities", ['Business', 'India', 'Technology']),
    ("Sports ministry approves funding for 2036 Olympics bid", ['Sports', 'India', 'Politics', 'Business']),
    ("Food delivery robot tested in Hyderabad technology district", ['Technology', 'India', 'Business']),
    ("Meditation app from India reaches top charts in 50 countries", ['Health', 'India', 'Technology', 'Business']),
    ("Actor donates crores to build hospital in native village", ['Entertainment', 'India', 'Health']),
    ("New study links air quality to student academic performance", ['Health', 'Science', 'India']),
    ("Rajya Sabha passes bill on right to internet access", ['Politics', 'India', 'Technology']),
    ("Tiger population in India rises to 3700 according to census", ['India', 'Science']),
    ("Cloud kitchen startup raises 100 crore in series B funding", ['Business', 'India', 'Technology']),
    ("Women's IPL season draws record viewership and sponsorship", ['Sports', 'India', 'Business', 'Entertainment']),
    ("India and UAE sign free trade agreement boosting exports", ['Business', 'India', 'Politics']),
    ("Scientist develops low-cost hearing aid using recycled materials", ['Science', 'Health', 'India', 'Technology']),
    ("Chess champion defends world title in 14-game match series", ['Sports', 'India']),
    ("Farmer uses drone technology to double crop yield", ['Technology', 'India', 'Business', 'Science']),

    # --- Extra 40 samples for better training ---
    ("India's space program reaches new milestone with successful launch", ['Science', 'India', 'Technology']),
    ("Stock exchange trading halted after sudden market crash", ['Business']),
    ("New fitness app goes viral with 5 million downloads overnight", ['Health', 'Technology', 'Business']),
    ("National boxing champion wins gold at Asian Games", ['Sports', 'India']),
    ("Opposition leader arrested on corruption charges by CBI", ['Politics', 'India']),
    ("Movie director wins best film award at international festival", ['Entertainment', 'India']),
    ("Scientists create new antibiotic to fight drug-resistant bacteria", ['Science', 'Health']),
    ("India exports record amount of software services this year", ['Business', 'India', 'Technology']),
    ("Badminton player secures Olympic quota after world ranking rise", ['Sports', 'India']),
    ("Parliament discusses bill to regulate artificial intelligence use", ['Politics', 'India', 'Technology']),
    ("Climate activists protest outside oil company headquarters", ['Politics', 'Science']),
    ("Streaming giant invests 500 million in Indian content creation", ['Entertainment', 'India', 'Business', 'Technology']),
    ("Research team develops cheap solar panel from local materials", ['Science', 'India', 'Technology', 'Business']),
    ("Flood relief operations underway in three coastal districts", ['India', 'Politics']),
    ("Teenagers win robotics competition representing India globally", ['Technology', 'India', 'Science']),
    ("Health ministry launches vaccination drive for children under 5", ['Health', 'India', 'Politics']),
    ("Legendary musician performs final world tour concert in Mumbai", ['Entertainment', 'India']),
    ("National highway project creates thousands of jobs in rural area", ['Business', 'India', 'Politics']),
    ("Doctors warn about rise in heart attacks among young adults", ['Health', 'Science']),
    ("Cricketer becomes brand ambassador for top multinational company", ['Sports', 'India', 'Business', 'Entertainment']),
    ("Space tourism company sells first tickets for suborbital flights", ['Science', 'Technology', 'Business']),
    ("Finance minister presents interim budget ahead of elections", ['Business', 'India', 'Politics']),
    ("Virtual reality used to train surgeons for complex operations", ['Technology', 'Health', 'Science']),
    ("Local politician wins by-election by narrow margin of votes", ['Politics', 'India']),
    ("Nutrition study shows Indian traditional food prevents diabetes", ['Health', 'India', 'Science']),
    ("Tech unicorn from Bengaluru files for IPO on stock exchange", ['Technology', 'Business', 'India']),
    ("India wins Davis Cup tennis tie against tough opposition", ['Sports', 'India']),
    ("Community radio station helps farmers with crop advisory service", ['India', 'Technology', 'Business']),
    ("Bharat Biotech launches affordable vaccine for neglected disease", ['Health', 'India', 'Science', 'Business']),
    ("Documentary filmmaker wins award for tribal conservation story", ['Entertainment', 'India', 'Science']),
    ("Cybercrime cell arrests gang running online investment fraud", ['Technology', 'India', 'Politics']),
    ("Electric bus fleet launched in state capital reducing pollution", ['Technology', 'India', 'Business', 'Politics']),
    ("Indian author wins Booker Prize for debut literary novel", ['Entertainment', 'India']),
    ("Scientist invents low-cost prosthetic limb using 3D printing", ['Science', 'Health', 'Technology', 'India']),
    ("Bollywood music composer receives lifetime achievement award", ['Entertainment', 'India']),
    ("Centre approves new medical college in underserved district", ['Health', 'India', 'Politics']),
    ("Global chip shortage affects smartphone production worldwide", ['Technology', 'Business']),
    ("Marathon runner from India qualifies for Paris Olympics", ['Sports', 'India']),
    ("State government bans single-use plastic across all cities", ['Politics', 'India', 'Science']),
    ("Agritech startup uses satellite data to predict crop disease", ['Technology', 'India', 'Business', 'Science']),
]


# ════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ════════════════════════════════════════════════════════════

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ════════════════════════════════════════════════════════════
# STEP 3 — BUILD & TRAIN MODELS
# ════════════════════════════════════════════════════════════

def build_dataset():
    texts  = [preprocess_text(item[0]) for item in RAW_DATA]
    labels = [item[1] for item in RAW_DATA]
    return texts, labels


def train_and_evaluate():
    print("\n" + "═"*60)
    print("   MULTI-LABEL TEXT CATEGORIZATION")
    print("═"*60)

    texts, labels = build_dataset()
    print(f"\n📊 Dataset     : {len(texts)} headlines")
    print(f"🏷  Categories  : {', '.join(CATEGORIES)}")

    # Binarize labels  (e.g. ['Sports','India'] → [0,0,0,0,0,0,1,1])
    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    Y   = mlb.fit_transform(labels)

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                 sublinear_tf=True)
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    print(f"\n🔀 Train / Test : {X_train.shape[0]} / {X_test.shape[0]} samples")

    # ── Models ───────────────────────────────────────────────
    models = {
        'Logistic Regression': OneVsRestClassifier(
            LogisticRegression(max_iter=2000, C=5.0, random_state=42)),
        'Linear SVM': OneVsRestClassifier(
            LinearSVC(max_iter=3000, C=0.5, random_state=42)),
        'Random Forest': OneVsRestClassifier(
            RandomForestClassifier(n_estimators=200, random_state=42)),
    }

    results   = {}
    all_history = {}

    print(f"\n{'Model':<25} {'Hamming↓':>10} {'F1-Micro':>10} {'F1-Macro':>10} {'Subset Acc':>12}")
    print("-"*68)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        hl    = hamming_loss(y_test, y_pred)
        f1_mi = f1_score(y_test, y_pred, average='micro', zero_division=0)
        f1_ma = f1_score(y_test, y_pred, average='macro', zero_division=0)
        acc   = accuracy_score(y_test, y_pred)

        print(f"{name:<25} {hl:>10.4f} {f1_mi:>10.4f} {f1_ma:>10.4f} {acc:>12.4f}")

        results[name] = {
            'hamming_loss':  round(hl, 4),
            'f1_micro':      round(f1_mi, 4),
            'f1_macro':      round(f1_ma, 4),
            'subset_accuracy': round(acc, 4),
        }

        # Per-category F1
        per_cat_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        results[name]['per_category_f1'] = {
            CATEGORIES[i]: round(float(per_cat_f1[i]), 4)
            for i in range(len(CATEGORIES))
        }

    # Best model = highest F1-micro
    best_name  = max(results, key=lambda n: results[n]['f1_micro'])
    best_model = models[best_name]
    print(f"\n🏆 Best Model  : {best_name}  (F1-micro = {results[best_name]['f1_micro']})")

    return best_model, vectorizer, mlb, results, best_name


# ════════════════════════════════════════════════════════════
# STEP 4 — PREDICT ON NEW TEXT
# ════════════════════════════════════════════════════════════

def predict(model, vectorizer, mlb, text):
    clean = preprocess_text(text)
    x     = vectorizer.transform([clean])
    y     = model.predict(x)
    labels = mlb.inverse_transform(y)[0]
    return list(labels)


# ════════════════════════════════════════════════════════════
# STEP 5 — MAIN
# ════════════════════════════════════════════════════════════

def main():
    best_model, vectorizer, mlb, results, best_name = train_and_evaluate()

    # Test predictions on new unseen headlines
    test_headlines = [
        "India beats Australia in cricket World Cup final to win championship",
        "New AI chip by Intel promises 10x faster machine learning training",
        "Government launches free health insurance for farmers across states",
        "Bollywood blockbuster earns 300 crores in opening weekend record",
        "ISRO Chandrayaan mission discovers water ice near lunar south pole",
        "Startup raises 500 crore for electric vehicle charging network India",
        "Yoga festival in Rishikesh attracts thousands of international visitors",
    ]

    print("\n" + "═"*60)
    print("   PREDICTIONS ON NEW HEADLINES")
    print("═"*60)

    predictions = []
    for headline in test_headlines:
        labels = predict(best_model, vectorizer, mlb, headline)
        print(f"\n📰 {headline}")
        print(f"   🏷  Labels : {', '.join(labels) if labels else 'None'}")
        predictions.append({'text': headline, 'predicted_labels': labels})

    # Save results
    output = {
        'model_results': results,
        'best_model': best_name,
        'categories': CATEGORIES,
        'sample_predictions': predictions,
        'dataset_size': len(RAW_DATA),
        'metrics_explanation': {
            'hamming_loss': 'Lower is better. Fraction of wrong labels.',
            'f1_micro':     'Overall F1 treating each label equally.',
            'f1_macro':     'Average F1 per category (good for imbalance).',
            'subset_accuracy': 'Exact match — all labels must be correct.'
        }
    }

    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "═"*60)
    print("   ✅ DONE — results.json saved")
    print("═"*60 + "\n")


if __name__ == '__main__':
    main()
# This line intentionally left blank
