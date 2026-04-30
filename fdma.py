import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import graphviz

from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn import tree
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,silhouette_score,roc_auc_score,precision_score,recall_score,f1_score,ConfusionMatrixDisplay,roc_curve,auc
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.calibration import CalibratedClassifierCV

#single unsupervised
def k_means(df,show):
    inertia=[]
    k_values=range(1,10)

    X=df[functional_col_lst].copy()
    X=pd.get_dummies(X,drop_first=True)
    X=X.fillna(0)

    y = df['Fraud_Label_Number']

    scaler=StandardScaler()
    X_Scaled=scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_Scaled)

    #find k--does not find one better
    '''
    for k in k_values:
        kmeans_findK=KMeans(n_clusters=k,random_state=33,n_init=10)
        kmeans_findK.fit(X_Scaled)
        inertia.append(kmeans_findK.inertia_)

    kLocator=KneeLocator(k_values,inertia,curve="convex",direction="decreasing")
    best_k=kLocator.elbow
    if(best_k==None):
        k=4
    else:
        k=best_k
    '''
    #hard code k to be 4
    k=4

    kmeans=KMeans(n_clusters=k,random_state=33)
    kmeans.fit(X_Scaled)
    df['Cluster']=kmeans.labels_

    distances = kmeans.transform(X_Scaled)
    kmeans_score = -np.min(distances, axis=1)

    fpr, tpr, thresholds = roc_curve(y, kmeans_score)
    roc_auc = roc_auc_score(y, kmeans_score)

    print("KMeans ROC-AUC:", roc_auc)

    fraud_stat=df.groupby('Cluster').agg(Total_Transactions=('Fraud_Label_Number','count'),Fraudulent_Transactions=('Fraud_Label_Number','sum'))

    fraud_stat['Fraud_Rate']=fraud_stat['Fraudulent_Transactions']/fraud_stat['Total_Transactions']
    print(fraud_stat.sort_values('Fraud_Rate',ascending=False))

    labels = kmeans.fit_predict(X)

    score = silhouette_score(X_Scaled, labels,sample_size=5000,random_state=33)
    print(f"score:{score}")

    plt.figure()
    plt.hist(labels, bins=np.arange(labels.max()+2)-0.5)
    plt.title("K-Means Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    if(show):
        plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=5)
    plt.title("PCA Projection of Fraud Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    if(show):
        plt.show()

    cluster_df = pd.DataFrame({"cluster": labels, "fraud": y})

    fraud_rate = cluster_df.groupby("cluster")["fraud"].mean()

    plt.figure()
    plt.bar(fraud_rate.index, fraud_rate.values)
    plt.title("Fraud Rate per Cluster (K-Means)")
    plt.xlabel("Cluster")
    plt.ylabel("Fraud Rate")
    if(show):
        plt.show()
    
    return fpr, tpr, roc_auc

#single supervised
def dec_tree(df,show):
    print("WIP")
    print(df)
    print("\n\n----\nThank you\n----\n\n")

    irrelevant=['Transaction_ID',
    'Customer_ID',
    'Merchant_ID',
    'Device_ID',
    'IP_Address',
    'Fraud_Label',
    'Transaction_Time'
    ]

    df_clean=df.drop(columns=irrelevant)

    catogory_columns=['Transaction_Type','Merchant_Category','Transaction_Location','Customer_Home_Location','Card_Type']

    y_n_columns=['Is_International_Transaction','Is_New_Merchant','Unusual_Time_Transaction']

    label_decoder={}

    for column in catogory_columns:
        labelEncoded=LabelEncoder()
        df_clean[column]=labelEncoded.fit_transform(df_clean[column].astype(str))
        label_decoder[column]=labelEncoded


    print(df['Transaction_Date'].head())

    df_clean['Transaction_Date']=pd.to_datetime(df_clean['Transaction_Date'],format='%m/%d/%y')
    df_clean['Day_Of_Week']=(df_clean['Transaction_Date'].dt.dayofweek)
    df_clean['Month']=(df_clean['Transaction_Date'].dt.month)
    df_clean['Is_Weekend']=df_clean['Day_Of_Week'].isin([5,6]).astype(int)
    df_clean=df_clean.drop(columns=['Transaction_Date'])
    

    df_clean['Amount_Bucket']=pd.qcut(
        df_clean['Transaction_Amount (in Million)'],q=5,
        labels=[0,1,2,3,4],duplicates='drop'
    ).cat.add_categories(-1).fillna(-1).astype(int)

    df_clean['Balance_Bucket']=pd.qcut(
        df_clean['Account_Balance (in Million)'],q=4,
        labels=[0,1,2,3],duplicates='drop'
    ).cat.add_categories(-1).fillna(-1).astype(int)


    df_clean['Distance_Bucket']=pd.cut(
        df_clean['Distance_From_Home'],bins=4,
        labels=[0,1,2,3]
    ).cat.add_categories(-1).fillna(-1).astype(int)

    df_clean['Max24hr_Bucket']=pd.qcut(
        df_clean['Max_Transaction_Last_24h (in Million)'],q=4,
        labels=[0,1,2,3]
    ).cat.add_categories(-1).fillna(-1).astype(int)

    df_clean=df_clean.drop(columns=[
        'Transaction_Amount (in Million)',
        'Account_Balance (in Million)',
        'Distance_From_Home',
        'Max_Transaction_Last_24h (in Million)'
    ])


    for column in y_n_columns:
        if(column in df_clean.columns):
            df_clean[column]=df_clean[column].map({'Yes':1,'No':0,True:1,False:0})
            df_clean[column]=df_clean[column].fillna(0).astype(int)

    print(df_clean.head())

    x=df_clean.drop(columns=['Fraud_Label_Number'])
    y=df_clean['Fraud_Label_Number']

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=.3,
        random_state=33,
        stratify=y
    )

    print(x_train.dtypes[x_train.dtypes == 'object'])
    
    clf=DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=33
    )

    scores=cross_val_score(clf,x,y,cv=5,scoring='f1')
    print("\n\n\n\n\n______HERERERERERERHERHEHRHERHERH______\n")
    print(scores)
    print(scores.mean())
    print(scores.std())

    clf.fit(x_train,y_train)


    y_proba = clf.predict_proba(x_test)[:, 1]

    thresholds = np.linspace(0.01, 0.5, 50)

    best_f1 = -1
    best_threshold = 0.5

    for t in thresholds:
        y_pred_temp = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred_temp)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    y_pred = (y_proba >= best_threshold).astype(int)

    print("Best Threshold:", best_threshold)
    print("Best F1:", best_f1)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    roc_auc=roc_auc_score(y_test,y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    print("\nAccuracy:  ", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    importance_df = pd.DataFrame({
        'Feature':   x.columns,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

    dot_data = tree.export_graphviz(clf,
        out_file=None,
        feature_names=x.columns,
        max_depth = 3,
        rotate=True,
        filled=True)

    if(show):
        graph = graphviz.Source(dot_data, format="png")
        graph.render("images/Decision_Tree")    
        graph.view()

    importances = clf.feature_importances_

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Legit','Fraud']
    )
    disp.plot()

    plt.title("Confusion Matrix (Decision Tree)")
    if(show):
        plt.show()


    plt.figure()
    plt.barh(x.columns, importances)
    plt.title("Decision Tree Feature Importance")
    if(show):
        plt.show()
    
    probs = clf.predict_proba(x_test)[:, 1]

    plt.figure()

    plt.hist(probs[y_test == 0], bins=30, alpha=0.6, label='Non-Fraud')
    plt.hist(probs[y_test == 1], bins=30, alpha=0.6, label='Fraud')

    plt.title("Decision Tree Fraud Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.axvline(best_threshold, linestyle='--', label='Threshold')
    plt.legend()

    if(show):
        plt.show()

    return fpr, tpr, roc_auc

#group supervised
def random_forest(df,show):

    X = df.drop(columns=['Fraud_Label_Number'])
    y = df['Fraud_Label_Number']

    high_cardinality_cols = [
        col for col in df.columns
        if df[col].dtype == 'object' and df[col].nunique() > 100
    ]

    print("Dropping high-cardinality columns:", high_cardinality_cols)

    leak_col=['Fraud_Label_Normal','Fraud_Label','Fraud','label','target','Previous_Fraud_Count']
    
    other_col=['Customer_ID','Transaction_ID','Device_ID','Merchant_ID']

    X = df.drop(columns=(
        high_cardinality_cols + 
        leak_col +
        other_col +
        ['Fraud_Label_Number'] + 
        ['Fraud_Label_Normal']), errors='ignore')

    print("Leakage check (should be 0 columns matching target):")
    print([col for col in X.columns if "Fraud" in col or "label" in col.lower()])

    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X = X.apply(pd.to_numeric, errors='coerce')

    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=33,
        stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=33,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    y_proba = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.1).astype(int)            

    print("rf_model")
    print(datetime.now())

    print("rf_model.ft")
    print(datetime.now())
    
    print("Class distribution in y_train:")
    print(y_train.value_counts())
    
    print("max:", y_proba.max())
    print("mean:", y_proba.mean())

    print("Max fraud probability:", y_proba.max())
    print("Min fraud probability:", y_proba.min())
    print("Mean fraud probability:", y_proba.mean())

    thresholds = np.linspace(0.05, 0.5, 20)
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== ROC-AUC Score ===")
    roc_auc=roc_auc_score(y_test, y_proba)
    print(roc_auc_score(y_test, y_proba))

    importances = rf_model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(feature_importance_df.head(10))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []

    count=0
    best_thresholds = []
    for train_index, test_index in kf.split(X, y):
        count+=1
        
        X_fold_train, X_fold_test = X.iloc[train_index], X.iloc[test_index]
        y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index]

        rf_model = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        calibrated_rf = CalibratedClassifierCV(
            rf_model,
            method='isotonic',
            cv=3
        )

        calibrated_rf.fit(X_fold_train, y_fold_train)

        y_proba = calibrated_rf.predict_proba(X_fold_test)[:, 1]

        thresholds = np.linspace(0.01, 0.5, 50)

        best_f1 = -1
        best_threshold = 0.5

        for t in thresholds:
            y_pred_temp = (y_proba >= t).astype(int)
            f1 = f1_score(y_fold_test, y_pred_temp)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        y_pred = (y_proba >= best_threshold).astype(int)
        best_thresholds.append(best_threshold)

        precision_list.append(precision_score(y_fold_test, y_pred))
        recall_list.append(recall_score(y_fold_test, y_pred))
        f1_list.append(f1_score(y_fold_test, y_pred))
        roc_auc_list.append(roc_auc_score(y_fold_test, y_proba))

    global_threshold = np.mean(best_thresholds)
    print("Global threshold (mean over folds):", global_threshold)

    y_proba_test = calibrated_rf.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= global_threshold).astype(int)

    print("\n=== FINAL TEST SET EVALUATION ===")
    print(classification_report(y_test, y_pred_test, zero_division=0))
    print(confusion_matrix(y_test, y_pred_test))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_test))

    print("=== K-Fold Cross Validation Results ===")
    print(f"Precision: {np.mean(precision_list):.4f}")
    print(f"Recall:    {np.mean(recall_list):.4f}")
    print(f"F1 Score:  {np.mean(f1_list):.4f}")
    print(f"ROC-AUC:   {np.mean(roc_auc_list):.4f}")


    fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
    roc_auc = roc_auc_score(y_test, y_proba_test)

    print("ROC-AUC:", roc_auc)

    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    y_proba_test = calibrated_rf.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= global_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred_test)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Legit','Fraud']
    )
    disp.plot()

    plt.title("Confusion Matrix (Random Forest)")
    plt.savefig("confusion_matrix.png")
    if(show):
        plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba_test)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("Random Forest ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if(show):
        plt.show()
    
    plt.figure()
    plt.hist(y_proba_test[y_test == 0], bins=30, alpha=0.6, label='Legit')
    plt.hist(y_proba_test[y_test == 1], bins=30, alpha=0.6, label='Fraud')
    plt.title("Random Forest Predicted Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.axvline(global_threshold, linestyle='--', label='Threshold')
    plt.legend()
    if(show):
        plt.show()

    return fpr, tpr, roc_auc

#group unsupervised
def isolation_forest(df,show):

    X = df.drop(columns=['Fraud_Label_Number'], errors='ignore')
    y = df['Fraud_Label_Number']

    high_cardinality_cols = [
        col for col in df.columns
        if df[col].dtype == 'object' and df[col].nunique() > 100
    ]

    leak_col = [
        'Fraud_Label_Normal','Fraud_Label','Fraud','label',
        'target','Previous_Fraud_Count'
    ]

    other_col = [
        'Customer_ID','Transaction_ID','Device_ID','Merchant_ID'
    ]

    X = df.drop(columns=(
        high_cardinality_cols +
        leak_col +
        other_col +
        ['Fraud_Label_Number']
    ), errors='ignore')

    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    print("Shape of X:", X.shape)

    iso_model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    iso_model.fit(X_train)

    y_pred_raw = iso_model.predict(X_test)
    
    scores = iso_model.decision_function(X_test)

    threshold = np.percentile(scores, 5)

    y_pred = (scores <= threshold).astype(int)

    y_scores = -iso_model.decision_function(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)

    print("\n=== ROC-AUC Score ===")
    print(roc_auc)

    print("\nPredicted fraud count:", sum(y_pred))

    print("\nScore stats:")
    print(np.min(scores), np.max(scores), np.mean(scores))

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Legit','Fraud']
    )
    disp.plot()

    plt.title("Confusion Matrix (Isolation Forest)")
    plt.savefig("confusion_matrix.png")
    if(True):
        plt.show()

    plt.figure()
    plt.hist(scores, bins=50)
    plt.title("Isolation Forest Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    if(show):
        plt.show()

    plt.figure()
    plt.hist(scores[y_test == 0], bins=50, alpha=0.5, label="Normal")
    plt.hist(scores[y_test == 1], bins=50, alpha=0.5, label="Fraud")
    plt.legend()
    plt.title("Isolation Forest Score Separation")
    if(show):
        plt.show() 
    
    return fpr, tpr, roc_auc


show=False
starttime=datetime.now()
print(f"start: {datetime.now()}")

df=pd.read_csv('~/Downloads/FraudShield_Banking_Data.csv')
df['Fraud_Label_Number']=df['Fraud_Label'].map({'Normal':0,'Fraud':1})

df=df.dropna(subset="Fraud_Label")

df_train,df_test=train_test_split(df,test_size=.3,random_state=33,stratify=df['Fraud_Label_Number'])

#Columns 9,9,7
#Transaction_ID,Customer_ID,Transaction_Amount,Transaction_Time,Transaction_Date,Transaction_Type,Merchant_ID,Merchant_Category,Transaction_Location
#Customer_Home_Location,Distance_From_Home,Device_ID,IP_Address,Card_Type,Account_Balance,Daily_Transaction_Count,Weekly_Transaction_Count,Avg_Transaction_amount
# Max_Transaction_Last_24h,Is_International_Transaction,Is_New_Merchant,Failed_Transaction_Count,Unusual_Time_Transaction,Previous_Fraud_Count,Fraud_Label

column_lst=["Transaction_ID","Customer_ID","Transaction_Amount (in Million)","Transaction_Time","Transaction_Date","Transaction_Type","Merchant_ID","Merchant_Category","Transaction_Location",
"Customer_Home_Location","Distance_From_Home","Device_ID","IP_Address","Card_Type","Account_Balance (in Million)","Daily_Transaction_Count","Weekly_Transaction_Count","Avg_Transaction_Amount (in Million)",
"Max_Transaction_Last_24h (in Million)","Is_International_Transaction","Is_New_Merchant","Failed_Transaction_Count","Unusual_Time_Transaction","Previous_Fraud_Count"]

functional_col_lst=["Transaction_Amount (in Million)","Transaction_Time","Transaction_Date","Transaction_Type","Merchant_Category","Transaction_Location",
"Customer_Home_Location","Distance_From_Home","Card_Type","Account_Balance (in Million)","Daily_Transaction_Count","Weekly_Transaction_Count","Avg_Transaction_Amount (in Million)",
"Max_Transaction_Last_24h (in Million)","Is_International_Transaction","Is_New_Merchant","Failed_Transaction_Count","Unusual_Time_Transaction","Previous_Fraud_Count"]


print(df.head())

fraud_count=df.groupby('Fraud_Label').count()
print(fraud_count)

print("\n-----------\nBEGINGINGING ANALYSIS\n-----------\n")
print(datetime.now()-starttime)

print("\n\n\n\n-----KMEANS-----\n\n")
print(datetime.now())

(fpr_km, tpr_km,kmeans_roc)=k_means(df,False)

print("\n\n\n\n-----Decision Tree-----\n\n")
print(datetime.now())

(fpr_dec, tpr_dec,dec_roc)=dec_tree(df,False)

print("\n\n\n\n-----Random Forest-----\n\n")
print(datetime.now())

(fpr_rf, tpr_rf,rforest_roc)=random_forest(df,False)

print("\n\n\n\n-----Isolation Forest-----\n\n")
print(datetime.now())

(fpr_if, tpr_if,iso_roc)=isolation_forest(df,False)

print("\n-----------\nENDING ANALYSIS\n-----------\n")
print(datetime.now())
print(f"Total Time:{datetime.now()-starttime}")


results = {
    "KMeans": kmeans_roc,
    "Decision Tree": dec_roc,
    "Random Forest": rforest_roc,
    "Isolation Forest": iso_roc
}

print("\nModel Ranking (by ROC-AUC):")
for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(model, score)


plt.figure()

# Random Forest
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rforest_roc:.3f})')

# Decision Tree
plt.plot(fpr_dec, tpr_dec, label=f'Decision Tree (AUC = {dec_roc:.3f})')

# KMeans
plt.plot(fpr_km, tpr_km, label=f'KMeans (AUC = {kmeans_roc:.3f})')

# Isolation Forest
plt.plot(fpr_if, tpr_if, label=f'Isolation Forest (AUC = {iso_roc:.3f})')

# Baseline
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison Across Models")
plt.legend()

if(True):
    plt.show()
