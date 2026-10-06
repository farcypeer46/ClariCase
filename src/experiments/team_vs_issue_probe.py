# Probe: team-first vs issue-first vs hierarchical (TF-IDF + LinearSVC on baseline splits).
# Run from repo root: python -m src.experiments.team_vs_issue_probe
import pandas as pd, numpy as np, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
P='data/processed/'
m=pd.read_csv(P+'issue_mapping.csv')
key=dict(zip(zip(m['product'],m['issue']),m.product_issue_id)); i2t=dict(zip(m.product_issue_id,m.team_id))
tr=pd.read_csv(P+'splits/train.csv',keep_default_na=False); te=pd.read_csv(P+'splits/test.csv',keep_default_na=False)
for d in (tr,te): d['pid']=[key.get(k) for k in zip(d['product'],d['issue'])]
tr=tr.dropna(subset=['pid']).reset_index(drop=True); te=te.dropna(subset=['pid']).reset_index(drop=True)
print(len(tr),len(te),tr.pid.nunique(),flush=True)
v=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000,sublinear_tf=True)
Xtr=v.fit_transform(tr.complaint_text); Xte=v.transform(te.complaint_text)
def mdl(): return LogisticRegression(C=5,max_iter=300,class_weight='balanced',solver='saga')
def svc(): return LinearSVC(C=0.5,class_weight='balanced')
t0=time.time()
A=svc().fit(Xtr,tr.team_id); pa=A.predict(Xte)
print('A team-direct            team acc %.3f mF1 %.3f'%(accuracy_score(te.team_id,pa),f1_score(te.team_id,pa,average='macro')),flush=True)
B=svc().fit(Xtr,tr.pid); SB=B.decision_function(Xte); cls=B.classes_
pb=cls[SB.argmax(1)]; tb=np.array([i2t[c] for c in pb])
print('B flat-73-issue          issue acc %.3f mF1 %.3f'%(accuracy_score(te.pid,pb),f1_score(te.pid,pb,average='macro')))
print('B -> team via mapping    team acc %.3f mF1 %.3f'%(accuracy_score(te.team_id,tb),f1_score(te.team_id,tb,average='macro')),flush=True)
teams=sorted(set(i2t.values()))
models={t:svc().fit(Xtr[(tr.team_id==t).values],tr.pid[tr.team_id==t]) for t in teams}
pc=np.empty(len(te),dtype=object); po=np.empty(len(te),dtype=object)
for t in teams:
    idx=np.where(pa==t)[0]; 
    if len(idx): pc[idx]=models[t].predict(Xte[idx])
    idx=np.where(te.team_id.values==t)[0]; po[idx]=models[t].predict(Xte[idx])
print('C hier team->issue       issue acc %.3f mF1 %.3f'%(accuracy_score(te.pid,pc),f1_score(te.pid,pc,average='macro')))
print('  (oracle team) issue    acc %.3f mF1 %.3f'%(accuracy_score(te.pid,po),f1_score(te.pid,po,average='macro')))
te['po']=po
print('within-team issue acc (oracle team):'); print((te.po==te.pid).groupby(te.team_id).mean().round(3).to_string())
print('secs',round(time.time()-t0))
