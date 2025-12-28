import pandas as pd
import numpy as np
import seaborn as sns
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

root = Tk()
root.title("Adult Income Analysis - KNN & Plotly")
root.geometry("1200x800")

df = None
df_clean = None
k_value = IntVar(value=5)
knn_model = None
scaler_model = None
label_encoders = {}
target_encoder = None
X_columns = None


left_panel = Frame(root, width=400, bg="#f0f0f0", padx=10, pady=10)
left_panel.pack(side="left", fill="y")

right_panel = Frame(root, bg="white")
right_panel.pack(side="right", fill="both", expand=True)


plot_canvas = Canvas(right_panel, bg="white")
plot_scrollbar = Scrollbar(right_panel, orient="vertical", command=plot_canvas.yview)
plot_scrollable_frame = Frame(plot_canvas, bg="white")

plot_scrollable_frame.bind(
    "<Configure>",
    lambda e: plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))
)

plot_canvas.create_window((0, 0), window=plot_scrollable_frame, anchor="nw")
plot_canvas.configure(yscrollcommand=plot_scrollbar.set)

plot_canvas.pack(side="top", fill="both", expand=True)
plot_scrollbar.pack(side="right", fill="y")

result_box = Text(right_panel, height=12, bg="#eef", font=("Courier", 10))
result_box.pack(side="bottom", fill="x", padx=10, pady=10)

def clear_plot():
    for widget in plot_scrollable_frame.winfo_children():
        widget.destroy()


def load_csv():
    global df
    path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if path:
        df = pd.read_csv(path)
        result_box.delete("1.0", END)
        result_box.insert(END, f"--- Data Overview ---\nShape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        result_box.insert(END, f"--- Head (First 5 rows) ---\n{df.head()}\n\n")
        result_box.insert(END, f"--- Tail (Last 5 rows) ---\n{df.tail()}\n\n")
        result_box.insert(END, f"--- Describe ---\n{df.describe()}\n\n")
        buffer = io.StringIO()
        df.info(buf=buffer)
        result_box.insert(END, f"--- Info ---\n{buffer.getvalue()}")
        messagebox.showinfo("Loaded", f"Data Loaded Successfully! Rows: {df.shape[0]}")

def clean_data():
    global df_clean
    if df is None:
        messagebox.showwarning("Error", "Load data first")
        return
    temp = df.copy()
    temp.replace('?', np.nan, inplace=True)
    important_cols = ['workclass', 'occupation', 'native.country']
    temp.dropna(subset=[c for c in important_cols if c in temp.columns], inplace=True)
    df_clean = temp.copy()
    messagebox.showinfo("Cleaned", f"Rows after cleaning: {df_clean.shape[0]}")

def remove_outliers():
    global df_clean
    if df_clean is None:
        messagebox.showwarning("Error", "Clean data first")
        return
    temp = df_clean.copy()
    cols = ['age', 'capital.gain', 'capital.loss', 'hours.per.week']
    for col in cols:
        if col in temp.columns:
            Q1 = temp[col].quantile(0.25)
            Q3 = temp[col].quantile(0.75)
            IQR = Q3 - Q1
            temp = temp[(temp[col] >= Q1 - 1.5 * IQR) & (temp[col] <= Q3 + 1.5 * IQR)]
    df_clean = temp.copy()
    messagebox.showinfo("Outliers Removed", f"Rows now: {df_clean.shape[0]}")

def save_cleaned_data():
    if df_clean is None:
        return
    path = filedialog.asksaveasfilename(defaultextension=".csv")
    if path:
        df_clean.to_csv(path, index=False)
        messagebox.showinfo("Saved", "Cleaned data saved")

def plot_numeric_columns_gui(data, title_prefix=""):
    if data is None:
        messagebox.showwarning("Error", "Load data first")
        return
    clear_plot()
    numeric_cols = data.select_dtypes(include='number').columns
    Label(plot_scrollable_frame, text=f"Static View: {title_prefix} Cleaning", font=("Arial", 12, "bold"), bg="white").pack(pady=10)
    for col in numeric_cols:
        fig = Figure(figsize=(12, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        sns.boxplot(x=data[col], ax=ax1, showfliers=(title_prefix == "Before"))
        ax1.set_title(f"{col} Box Plot", fontsize=12)
        sns.histplot(data[col], kde=True, ax=ax2)
        ax2.set_title(f"{col} Histogram", fontsize=12)
        canvas = FigureCanvasTkAgg(fig, plot_scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", pady=5)


def plot_with_plotly(data, title_prefix=""):
    if data is None:
        messagebox.showwarning("Error", "Load data first")
        return
    numeric_cols = data.select_dtypes(include='number').columns
    fig = make_subplots(rows=len(numeric_cols), cols=2,
                        subplot_titles=[f"{c} (Boxplot)" if i % 2 == 0 else f"{c} (Histogram)" for c in numeric_cols for i in range(2)])
    for i, col in enumerate(numeric_cols):
        fig.add_trace(go.Box(x=data[col], name=col), row=i + 1, col=1)
        fig.add_trace(go.Histogram(x=data[col], name=col), row=i + 1, col=2)
    fig.update_layout(height=300*len(numeric_cols), title_text=f"Interactive Analysis: {title_prefix}", showlegend=False)
    fig.show()

def run_classification_knn():
    global knn_model, scaler_model, label_encoders, target_encoder, X_columns
    if df_clean is None:
        messagebox.showwarning("Warning", "Clean data first")
        return
    target_col = [c for c in df_clean.columns if 'income' in c.lower()][0]
    X = df_clean.drop(target_col, axis=1)
    y = df_clean[target_col]
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    X_encoded = X.copy()
    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    scaler_model = StandardScaler()
    X_scaled = scaler_model.fit_transform(X_encoded)
    X_columns = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=k_value.get())
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    result_box.delete("1.0", END)
    result_box.insert(END, f"Model Trained!\nAccuracy: {accuracy_score(y_test, y_pred):.2f}\n\n{classification_report(y_test, y_pred)}")

def open_predict_window():
    if knn_model is None:
        messagebox.showerror("Error", "Please run the KNN model first!")
        return
    predict_win = Toplevel(root)
    predict_win.title("Make Prediction")
    predict_win.geometry("500x600")
    inputs = {}
    main_scroll = Frame(predict_win)
    main_scroll.pack(fill=BOTH, expand=1)
    canvas = Canvas(main_scroll)
    canvas.pack(side=LEFT, fill=BOTH, expand=1)
    scroll = Scrollbar(main_scroll, orient=VERTICAL, command=canvas.yview)
    scroll.pack(side=RIGHT, fill=Y)
    canvas.configure(yscrollcommand=scroll.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    inner_frame = Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")
    for col in X_columns:
        Label(inner_frame, text=col + ":").pack(pady=2)
        if col in label_encoders:
            val_var = StringVar()
            options = list(label_encoders[col].classes_)
            combo = ttk.Combobox(inner_frame, textvariable=val_var, values=options, state="readonly")
            combo.current(0)
            combo.pack(pady=2)
            inputs[col] = val_var
        else:
            val_var = StringVar(value="0")
            Entry(inner_frame, textvariable=val_var).pack(pady=2)
            inputs[col] = val_var
    def predict_result():
        try:
            input_data = []
            for col in X_columns:
                val = inputs[col].get()
                if col in label_encoders:
                    val = label_encoders[col].transform([val])[0]
                else:
                    val = float(val)
                input_data.append(val)
            input_scaled = scaler_model.transform([input_data])
            prediction = knn_model.predict(input_scaled)
            final_label = target_encoder.inverse_transform(prediction)[0]
            messagebox.showinfo("Prediction Result", f"The predicted income is: {final_label}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
    Button(predict_win, text="Predict Now", command=predict_result, bg="green", fg="white", height=2).pack(side="bottom", fill="x")

Label(left_panel, text="MAIN CONTROLS", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=5)
btn = {"width": 30}
Button(left_panel, text="1. Load CSV & Show Info", command=load_csv, bg="#d1ffbd", **btn).pack(pady=2)
Button(left_panel, text="2. Clean Data", command=clean_data, **btn).pack(pady=2)
Button(left_panel, text="3. Remove Outliers", command=remove_outliers, **btn).pack(pady=2)
Button(left_panel, text="4. Save Cleaned Data", command=save_cleaned_data, **btn).pack(pady=2)
Button(left_panel, text="5. Plot Before", command=lambda: plot_numeric_columns_gui(df, "Before"), **btn).pack(pady=2)
Button(left_panel, text="6. Plot After", command=lambda: plot_numeric_columns_gui(df_clean, "After"), **btn).pack(pady=2)
Button(left_panel, text="Plotly Interactive (Before)", command=lambda: plot_with_plotly(df, "Before"), bg="#ffe4b5", **btn).pack(pady=5)
Button(left_panel, text="Plotly Interactive (After)", command=lambda: plot_with_plotly(df_clean, "After"), bg="#ffe4b5", **btn).pack(pady=2)
Label(left_panel, text="K Neighbors").pack()
Entry(left_panel, textvariable=k_value, width=5).pack()
Button(left_panel, text="7. Run KNN Model", command=run_classification_knn, bg="#add8e6", **btn).pack(pady=5)
Button(left_panel, text="8. PREDICT NEW DATA", command=open_predict_window, bg="#90ee90", font=("Arial", 10, "bold"), **btn).pack(pady=10)

root.mainloop()