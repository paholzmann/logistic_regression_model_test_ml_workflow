import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore

class Main:
    def create_df(self, n):
        df = pd.DataFrame({
            "MitarbeiterID": range(1, n+1),
            "Name": [f"Mitarbeiter_{i}" for i in range(1, n+1)],
            "Abteilung": np.random.choice(["IT", "HR", "Sales", "Marketing"], n),
            "Alter": np.random.randint(18, 60, n),
            "Gehalt": np.random.choice([np.nan, * range(3000, 9000, 500)], n),
            "Dienstjahre": np.random.randint(0, 20, n),
            "Letzte_Bewertung": np.random.randint(1, 10, n),
            "Projektzeit": np.random.randint(10, 50, n),
            "Homeoffice": np.random.choice(["Ja", "Nein"], n),
            "Bonus": np.random.choice([np.nan, 1000, 2000, 3000, 4000], n)
        })
        return df
    
    def read_df(self, df):
        print(df.head())
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns}")

    def fill_missing_values(self, df):
        numeric_columns = ["Alter", "Gehalt", "Dienstjahre", "Letzte_Bewertung", "Projektzeit"]
        fill_with_zero_columns = ["Bonus"]
        non_numeric_columns = ["Abteilung", "Homeoffice"]
        for column in df.columns:
            if column in numeric_columns:
                df[column] = df[column].fillna(df[column].mean(skipna=True))
            elif column in fill_with_zero_columns:
                df[column] = df[column].fillna(0)
            elif column in non_numeric_columns:
                df[column] = df[column].replace(["IT", "HR", "Sales", "Marketing", "Ja", "Nein"], [0, 1, 2, 3, 1, 0])
        return df
    
    def add_new_features(self, df):
        df["Alter_Normalisiert"] = (df["Alter"] - df["Alter"].mean()) / df["Alter"].std()
        df["Bonus_pro_Dienstjahr"] = df["Bonus"] / (df["Dienstjahre"] + 1)
        df["Gehalt_pro_Projektstunde"] = df["Gehalt"] / df["Projektzeit"]
        return df
    
    def filter_aggregation(self, df):
        it_filtered_df = df.loc[(df["Abteilung"] == 0) & (df["Homeoffice"] == 0)  & (df["Letzte_Bewertung"] >= 4)]
        avg_salary_by_department = df.groupby("Abteilung")[["Gehalt", "Bonus"]].mean()
        n_largest_df = df.nlargest(3, "Bonus_pro_Dienstjahr")
        return it_filtered_df, avg_salary_by_department, n_largest_df
    
    def sort_and_rank(self, df):
        sorted_df = df.sort_values(by=["Gehalt_pro_Projektstunde"], ascending=False)
        sorted_df["Gehalt_Rang"] = sorted_df["Gehalt"].rank(ascending=False, method="dense")
        return sorted_df
    
    def merge_dfs(self, df, n):
        second_df = pd.DataFrame({
            "MitarbeiterID": range(1, n+1),
            "Weiterbildungsstunden": np.random.randint(0, 20, n)
        })
        merged_df = df.merge(second_df)
        merged_df["Projektzeit_pro_Weiterbildungsstunde"] = (merged_df["Projektzeit"] / merged_df["Weiterbildungsstunden"] + 1)
        return merged_df
    
    def groupby_department(self, df):
        avg_by_department = df.groupby("Abteilung")[["Alter", "Bonus", "Projektzeit"]].mean()
        return avg_by_department
    
    def complex_conditions(self, df):
        df = df.loc[((df["Letzte_Bewertung"] == 5) | (df["Bonus"] > 2000)) & (df["Gehalt_pro_Projektstunde"] > 150)]
        return df
    
    def prepare_for_model(self, df):
        target = "Abteilung"
        train_percent = 80
        train_nums = int((len(df) / 100) * train_percent)
        features = [column for column in df.columns if column != target]
        train_df = df.sample(n=train_nums, random_state=42)
        test_df = df.drop(train_df.index)
        x_train = train_df.drop(columns=[target, "MitarbeiterID", "Name"])
        y_train = train_df[target]
        x_test = test_df.drop(columns=[target, "MitarbeiterID", "Name"])
        y_test = test_df[target]


        return x_train, y_train, x_test, y_test
        
    def train_model(self, x_train, y_train, x_test, y_test):
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main = Main()
    n = 5000
    df = main.create_df(n=n)
    df = main.fill_missing_values(df=df)
    df = main.add_new_features(df=df)
    it_filtered_df, avg_salary_by_department, n_largest_df = main.filter_aggregation(df=df)
    sorted_df = main.sort_and_rank(df=df)
    merged_df = main.merge_dfs(df=df, n=n)
    avg_by_department = main.groupby_department(df=df)
    df = main.complex_conditions(df=df)
    x_train, y_train, x_test, y_test = main.prepare_for_model(df=df)
    main.train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
