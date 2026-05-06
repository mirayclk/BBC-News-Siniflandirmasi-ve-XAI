"""
BBC News Classification - Final Project
=========================================
Adımlar:
1. Veri Ön İşleme (ham vs işlenmiş karşılaştırması)
2. Model Eğitimi: TF-IDF + Logistic Regression (ham & işlenmiş)
3. Metrikler: F1-score, Accuracy, Classification Report
4. XAI: LIME ile yorumlanabilirlik
5. Görselleştirme & Analiz
"""

import os, warnings, re, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME yok → pip install lime")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ─── Stopwords (built-in, network gerekmez) ──────────────────────────────────
STOPWORDS_EN = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o","re",
    "ve","y","said","also","mr","mrs","would","could","one","two","three","new",
    "year","last","first","may","us","uk","told","make","made","get","got","use",
    "used","since","after","during","while","however","although","still","much",
    "many","time","way","day","back","well","even","come","take","put","set",
    "say","go","see","think","know","want","need","look","people","week","long",
    "later","over","per","cent","million","billion","government","company"
}

# Basit suffix-based lemmatizer (NLTK gerektirmez)
def simple_lemmatize(word):
    suffixes = [("ies","y"),("ied","y"),("ing",""),("ness",""),("tion","te"),
                ("ations","ate"),("ers","er"),("ly",""),("ments","ment"),
                ("ful",""),("less","")]
    for suf, rep in suffixes:
        if word.endswith(suf) and len(word)-len(suf) > 2:
            return word[:-len(suf)] + rep
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word

# ─── Ayarlar ─────────────────────────────────────────────────────────────────
DATA_PATH    = "bbc-news-data.csv"
OUT_DIR      = "output_figures"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)
PALETTE   = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]
CAT_ORDER = ["business","entertainment","politics","sport","tech"]

plt.rcParams.update({"figure.dpi":120,"font.size":11,
                     "axes.spines.top":False,"axes.spines.right":False})

# ─── 1. VERİ YÜKLEME ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  BBC NEWS SINIFLANDIRMA PROJESİ")
print("="*65)
df = pd.read_csv(DATA_PATH, sep="\t")
print(f"\n[1] Veri: {df.shape[0]} satır, {df.shape[1]} sütun")
print(f"    Kategoriler: {df['category'].value_counts().to_dict()}")

# ─── 2. ÖN İŞLEME ────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_EN and len(t) > 2]
    tokens = [simple_lemmatize(t) for t in tokens]
    return " ".join(tokens)

print("\n[2] Ön işleme uygulanıyor...")
t0 = time.time()
df["content_raw"]   = df["title"] + " " + df["content"]
df["content_clean"] = df["content_raw"].apply(preprocess)
print(f"    Tamamlandı ({time.time()-t0:.1f}s)")
df["len_raw"]   = df["content_raw"].str.split().str.len()
df["len_clean"] = df["content_clean"].str.split().str.len()
print(f"    Ortalama kelime (ham/temiz): {df['len_raw'].mean():.0f} / {df['len_clean'].mean():.0f}")
print(f"    Azalma: %{(1-df['len_clean'].mean()/df['len_raw'].mean())*100:.1f}")

# ─── 3. ENCODE & SPLIT ───────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(df["category"])
X_raw   = df["content_raw"].values
X_clean = df["content_clean"].values

(X_raw_tr, X_raw_te,
 X_cln_tr, X_cln_te,
 y_tr, y_te) = train_test_split(
    X_raw, X_clean, y,
    test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"\n[3] Train/Test: {len(y_tr)} / {len(y_te)}")

# ─── 4. PIPELINE ─────────────────────────────────────────────────────────────
def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs",
                                   random_state=RANDOM_STATE))
    ])

print("\n[4] Model eğitimi...")
pipe_raw = build_pipeline(); pipe_raw.fit(X_raw_tr, y_tr)
pred_raw = pipe_raw.predict(X_raw_te); prob_raw = pipe_raw.predict_proba(X_raw_te)
pipe_cln = build_pipeline(); pipe_cln.fit(X_cln_tr, y_tr)
pred_cln = pipe_cln.predict(X_cln_te); prob_cln = pipe_cln.predict_proba(X_cln_te)
print("    Eğitim tamamlandı.")

# ─── 5. METRİKLER ────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, label):
    y_bin = label_binarize(y_true, classes=list(range(5)))
    return {
        "Model": label,
        "Accuracy":     accuracy_score(y_true, y_pred),
        "F1-Macro":     f1_score(y_true, y_pred, average="macro"),
        "F1-Weighted":  f1_score(y_true, y_pred, average="weighted"),
        "AUC-ROC":      roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    }

res_raw = compute_metrics(y_te, pred_raw, prob_raw, "Ham Veri")
res_cln = compute_metrics(y_te, pred_cln, prob_cln, "Ön İşlenmiş")
results_df = pd.DataFrame([res_raw, res_cln]).set_index("Model")

print("\n[5] METRİKLER")
print(results_df.to_string(float_format="{:.4f}".format))
print("\n--- Sınıf Raporu (Ham Veri) ---")
print(classification_report(y_te, pred_raw, target_names=le.classes_))
print("--- Sınıf Raporu (Ön İşlenmiş) ---")
print(classification_report(y_te, pred_cln, target_names=le.classes_))

print("\n[5b] Cross-Validation (5-fold)...")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_raw = cross_val_score(pipe_raw, X_raw,   y, cv=skf, scoring="f1_macro")
cv_cln = cross_val_score(pipe_cln, X_clean, y, cv=skf, scoring="f1_macro")
print(f"    Ham      CV F1: {cv_raw.mean():.4f} ± {cv_raw.std():.4f}")
print(f"    İşlenmiş CV F1: {cv_cln.mean():.4f} ± {cv_cln.std():.4f}")

# ─── 6. GÖRSELLEŞTİRME ──────────────────────────────────────────────────────
print("\n[6] Grafikler oluşturuluyor...")

# 6-A: Veri dağılımı
fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle("Veri Seti Analizi", fontsize=14, fontweight="bold")
counts = df["category"].value_counts()[CAT_ORDER]
axes[0].bar(CAT_ORDER, counts.values, color=PALETTE, edgecolor="white")
axes[0].set_title("Kategori Dağılımı"); axes[0].set_ylabel("Haber Sayısı")
for i,v in enumerate(counts.values): axes[0].text(i, v+5, str(v), ha="center", fontsize=10)
bp = axes[1].boxplot([df[df["category"]==c]["len_raw"].values for c in CAT_ORDER],
                     labels=CAT_ORDER, patch_artist=True,
                     boxprops=dict(facecolor="#AEC6E8"), medianprops=dict(color="navy",linewidth=2))
axes[1].set_title("Kelime Sayısı (Ham)"); axes[1].set_ylabel("Kelime Sayısı")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/01_data_distribution.png", bbox_inches="tight"); plt.close()

# 6-B: Preprocessing histogram
fig, axes = plt.subplots(1, 2, figsize=(13,5))
fig.suptitle("Ön İşleme: Kelime Sayısı Karşılaştırması", fontsize=14, fontweight="bold")
axes[0].hist(df["len_raw"],   bins=50, color="#4C72B0", alpha=0.75, label="Ham", edgecolor="white")
axes[0].hist(df["len_clean"], bins=50, color="#DD8452", alpha=0.75, label="Temiz", edgecolor="white")
axes[0].set_xlabel("Kelime"); axes[0].set_ylabel("Frekans"); axes[0].legend()
axes[0].set_title("Kelime Dağılımı (Ham vs Temiz)")
avg_raw   = [df[df["category"]==c]["len_raw"].mean()   for c in CAT_ORDER]
avg_clean = [df[df["category"]==c]["len_clean"].mean() for c in CAT_ORDER]
x = np.arange(len(CAT_ORDER)); w=0.35
axes[1].bar(x-w/2, avg_raw,   w, label="Ham",   color="#4C72B0", alpha=0.85, edgecolor="white")
axes[1].bar(x+w/2, avg_clean, w, label="Temiz", color="#DD8452", alpha=0.85, edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(CAT_ORDER)
axes[1].set_title("Ort. Kelime/Kategori"); axes[1].set_ylabel("Kelime"); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/02_preprocessing_comparison.png", bbox_inches="tight"); plt.close()

# 6-C: Metrik karşılaştırma
fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle("Model Performansı: Ham vs Ön İşlenmiş", fontsize=14, fontweight="bold")
metric_cols = ["Accuracy","F1-Macro","F1-Weighted","AUC-ROC"]
x=np.arange(len(metric_cols)); w=0.35
bars1 = axes[0].bar(x-w/2, results_df.loc["Ham Veri",metric_cols],    w, label="Ham Veri",     color="#4C72B0", edgecolor="white")
bars2 = axes[0].bar(x+w/2, results_df.loc["Ön İşlenmiş",metric_cols], w, label="Ön İşlenmiş", color="#DD8452", edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(metric_cols, fontsize=9)
axes[0].set_ylim(0.85,1.01); axes[0].set_ylabel("Skor"); axes[0].legend(); axes[0].set_title("Genel Metrikler")
for bar in list(bars1)+list(bars2):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
bp = axes[1].boxplot([cv_raw, cv_cln], patch_artist=True, labels=["Ham","İşlenmiş"], widths=0.4)
for patch,color in zip(bp["boxes"],["#4C72B0","#DD8452"]): patch.set_facecolor(color); patch.set_alpha(0.7)
for med in bp["medians"]: med.set(color="navy",linewidth=2)
axes[1].set_ylabel("F1-Macro (5-Fold CV)"); axes[1].set_title("Cross-Validation"); axes[1].set_ylim(0.85,1.01)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/03_metric_comparison.png", bbox_inches="tight"); plt.close()

# 6-D: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(15,6))
fig.suptitle("Karışıklık Matrisleri", fontsize=14, fontweight="bold")
for ax, pred, title in zip(axes, [pred_raw,pred_cln], ["Ham Veri","Ön İşlenmiş Veri"]):
    cm = confusion_matrix(y_te, pred)
    cm_norm = cm.astype(float)/cm.sum(axis=1,keepdims=True)
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax, linewidths=0.5, annot_kws={"size":11})
    ax.set_title(f"{title}\nAcc={accuracy_score(y_te,pred):.4f}  F1={f1_score(y_te,pred,average='macro'):.4f}")
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/04_confusion_matrices.png", bbox_inches="tight"); plt.close()

# 6-E: Per-class F1
f1_raw = f1_score(y_te, pred_raw, average=None)
f1_cln = f1_score(y_te, pred_cln, average=None)
fig, ax = plt.subplots(figsize=(10,5))
x=np.arange(len(le.classes_)); w=0.35
b1=ax.bar(x-w/2, f1_raw, w, label="Ham",     color="#4C72B0", edgecolor="white", alpha=0.85)
b2=ax.bar(x+w/2, f1_cln, w, label="İşlenmiş", color="#DD8452", edgecolor="white", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(le.classes_)
ax.set_ylim(0.8,1.01); ax.set_ylabel("F1-Score"); ax.legend()
ax.set_title("Sınıf Başına F1-Score", fontsize=13, fontweight="bold")
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/05_per_class_f1.png", bbox_inches="tight"); plt.close()

# 6-F: ROC curves
fig, axes = plt.subplots(1, 2, figsize=(15,6))
fig.suptitle("ROC Eğrileri (One-vs-Rest)", fontsize=14, fontweight="bold")
y_bin = label_binarize(y_te, classes=list(range(5)))
for ax, prob, title in zip(axes, [prob_raw,prob_cln], ["Ham Veri","Ön İşlenmiş"]):
    for i, cls in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:,i], prob[:,i])
        auc = roc_auc_score(y_bin[:,i], prob[:,i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})", color=PALETTE[i], linewidth=2)
    ax.plot([0,1],[0,1],"k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/06_roc_curves.png", bbox_inches="tight"); plt.close()

# 6-G: Top TF-IDF features
tfidf = pipe_cln.named_steps["tfidf"]
clf   = pipe_cln.named_steps["clf"]
fig, axes = plt.subplots(1, 5, figsize=(22,5))
fig.suptitle("En Önemli 15 TF-IDF Özelliği (İşlenmiş Model)", fontsize=13, fontweight="bold")
for ax, cls_idx, cat in zip(axes, range(5), le.classes_):
    coef = clf.coef_[cls_idx]
    top_idx   = np.argsort(coef)[-15:][::-1]
    top_words = [tfidf.get_feature_names_out()[i] for i in top_idx]
    top_vals  = coef[top_idx]
    ax.barh(top_words[::-1], top_vals[::-1], color=PALETTE[cls_idx], edgecolor="white", alpha=0.85)
    ax.set_title(cat.upper()); ax.set_xlabel("Koef."); ax.tick_params(labelsize=8)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/07_top_features.png", bbox_inches="tight"); plt.close()

# 6-H: WordCloud (opsiyonel)
if WORDCLOUD_AVAILABLE:
    fig, axes = plt.subplots(1, 5, figsize=(20,4))
    fig.suptitle("Kategori Bazlı Word Cloud (Ön İşlenmiş)", fontsize=13, fontweight="bold")
    for ax, cat, col in zip(axes, CAT_ORDER, PALETTE):
        text = " ".join(df[df["category"]==cat]["content_clean"])
        wc = WordCloud(width=300, height=250, background_color="white",
                       color_func=lambda *a,**k: col, max_words=80).generate(text)
        ax.imshow(wc, interpolation="bilinear"); ax.set_title(cat.upper()); ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/08_wordclouds.png", bbox_inches="tight"); plt.close()

# ─── 7. XAI: LIME ─────────────────────────────────────────────────────────────
print("\n[7] XAI - LIME...")
if LIME_AVAILABLE:
    explainer = LimeTextExplainer(class_names=le.classes_, random_state=RANDOM_STATE)

    def predict_fn_clean(texts):
        cleaned = [preprocess(t) for t in texts]
        return pipe_cln.predict_proba(cleaned)

    lime_results = {}
    for cat in CAT_ORDER:
        idx   = np.where(y_te == le.transform([cat])[0])[0][0]
        text  = X_raw_te[idx]
        true  = le.classes_[y_te[idx]]
        pred  = le.classes_[pipe_cln.predict([preprocess(text)])[0]]
        cls_id = le.transform([cat])[0]
        exp   = explainer.explain_instance(text, predict_fn_clean,
                                            num_features=10, num_samples=300, labels=[cls_id])
        lime_results[cat] = {"exp":exp,"true":true,"pred":pred,"cls_id":cls_id}
        print(f"    {cat:15s} → Gerçek:{true:15s} Tahmin:{pred}")

    fig, axes = plt.subplots(1, 5, figsize=(22,6))
    fig.suptitle("XAI - LIME Açıklamaları (Kategori Başına 1 Örnek)", fontsize=13, fontweight="bold")
    for ax, cat in zip(axes, CAT_ORDER):
        r    = lime_results[cat]
        vals = r["exp"].as_list(label=r["cls_id"])
        words  = [v[0] for v in vals]
        scores = [v[1] for v in vals]
        colors = ["#4C72B0" if s>0 else "#C44E52" for s in scores]
        ax.barh(words[::-1], scores[::-1], color=colors[::-1], edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{cat.upper()}\nG:{r['true']} | T:{r['pred']}", fontsize=9)
        ax.set_xlabel("LIME", fontsize=8); ax.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/09_lime_explanations.png", bbox_inches="tight"); plt.close()

    # Yanlış tahmin analizi
    wrong_mask = (pred_cln != y_te)
    if wrong_mask.sum() > 0:
        wi    = np.where(wrong_mask)[0][0]
        wtext = X_raw_te[wi]
        wtrue = le.classes_[y_te[wi]]
        wpred = le.classes_[pred_cln[wi]]
        wpred_id = le.transform([wpred])[0]
        exp_w = explainer.explain_instance(wtext, predict_fn_clean,
                                            num_features=10, num_samples=300, labels=[wpred_id])
        vals   = exp_w.as_list(label=wpred_id)
        words  = [v[0] for v in vals]
        scores = [v[1] for v in vals]
        colors = ["#4C72B0" if s>0 else "#C44E52" for s in scores]
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(words[::-1], scores[::-1], color=colors[::-1], edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"LIME - Yanlış Sınıflandırma\nGerçek: {wtrue}  |  Tahmin: {wpred}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("LIME Katkısı")
        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/10_lime_wrong_prediction.png", bbox_inches="tight"); plt.close()
        print(f"    Yanlış tahmin → Gerçek:{wtrue}  Tahmin:{wpred}")
else:
    print("    LIME bulunamadı.")

# ─── 8. SONUÇ ────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SONUÇ ÖZETİ")
print("="*65)
print(results_df.to_string(float_format="{:.4f}".format))
print(f"\nCV (Ham)      : {cv_raw.mean():.4f} ± {cv_raw.std():.4f}")
print(f"CV (İşlenmiş) : {cv_cln.mean():.4f} ± {cv_cln.std():.4f}")
diff = results_df.loc["Ön İşlenmiş","F1-Macro"] - results_df.loc["Ham Veri","F1-Macro"]
print(f"\nÖn işleme etkisi (F1-Macro): {'+'if diff>=0 else ''}{diff:.4f}")
print(f"\nKaydedilen grafikler ({OUT_DIR}/):")
for f in sorted(os.listdir(OUT_DIR)): print(f"  {f}")
print("\n[TAMAMLANDI]\n")
