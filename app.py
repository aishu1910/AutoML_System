import streamlit as st
import preprocessing as pp
import model_regression as rm
import model_classification as cm
import pandas as pd
import joblib
import numpy as np
from styles import load_css

st.markdown(load_css(), unsafe_allow_html=True)

st.title("The AutoML System")
# 1. Initialize the step in session state if it doesn't exist
if 'step' not in st.session_state:
    st.session_state.step = 1

# 2. Create the horizontal navigation bar
# We use more columns to create spacing
cols = st.columns([1, 1, 1, 1, 1])

# Step 1 Button
if cols[0].button("📁 Upload"):
    st.session_state.step = 1

# Step 2 Button
if cols[1].button("⚙️ Configure"):
    st.session_state.step = 2

# Step 3 Button
if cols[2].button("🔄 Preprocess"):
    st.session_state.step = 3

# Step 4 Button
if cols[3].button("📊 Evaluate"):
    st.session_state.step = 4

# Step 5 Button
if cols[4].button("📤 Export"):
    st.session_state.step = 5

st.divider()

# 3. Display different content based on the active step
if st.session_state.step == 1:
    file = st.file_uploader(
        "Upload your dataset",
        label_visibility="collapsed"
    )

    if file is not None:
        st.session_state.df = pp.load_data(file)
        if isinstance(st.session_state.df, pd.DataFrame):
            st.success("File loaded successfully!")
            st.dataframe(st.session_state.df.head())
        else:
            st.error(st.session_state.df)

elif st.session_state.step == 2:
    st.header("Configure Model")
    df = st.session_state.df

    # 1. Target Variable Selection
    st.subheader("1. Identify Target")
    target_col = st.selectbox("Which column do you want to predict?", df.columns)
    st.session_state.target = target_col

    # 2. Suggested Drops (The "Smart" part)
    st.subheader("2. Clean Columns")
    # Identify columns that look like IDs or have 'name' in them
    suggested_drops = [col for col in df.columns if 'id' in col.lower() or 'name' in col.lower()]

    cols_to_drop = st.multiselect(
        "Select columns to remove (we suggested some based on names):",
        options=df.columns,
        default=suggested_drops
    )
    st.session_state.drops = cols_to_drop

    # 3. Scaling Choice
    st.subheader("3. Scaling")
    scale_choice = st.radio(
        "Do you want to scale your numerical data?",
        options=["Yes (Recommended for KNN/SVM/Linear)", "No (Best for Trees/XGBoost)"]
    )
    # Convert choice to 0 or 1 for your preprocess function
    st.session_state.scale = 1 if "Yes" in scale_choice else 0

    # 4. Final Confirmation Button
    if st.button("Finalize Configuration & Preprocess"):
        # Apply the drops here in app.py before sending to preprocess
        cleaned_df = df.drop(columns=st.session_state.drops)

        # Call your preprocess function!
        # Assuming pp.preprocess returns (X_train, X_test, y_train, y_test)
        results = pp.preprocess(cleaned_df, st.session_state.target, st.session_state.scale)

        # Save these for the next step (Model Training)
        st.session_state.processed_data = results
        st.session_state.step = 3  # Move to the next circle!
        st.rerun()

elif st.session_state.step == 3:
    st.header("Step 3: Data Preprocessing")

    st.write("We are ready to clean, encode, and scale your data based on your configurations.")

    # 1. Start the process with a button
    if st.button("Run Preprocessing Engine"):

        # 2. Use the 'with st.spinner' for the loading effect
        with st.spinner("🤖 Applying math, handling nulls, and encoding strings..."):

            # 3. Retrieve the variables from session_state
            # (Make sure these names match what you saved in Step 2!)
            raw_data = st.session_state.df
            target = st.session_state.target
            scale_val = st.session_state.scale
            drops = st.session_state.drops

            # Apply the user's drops first
            df_to_process = raw_data.drop(columns=drops)

            # 4. Call your function from preprocess.py
            # This returns the (X_train, X_test, y_train, y_test) tuple
            try:
                processed_results = pp.preprocess(df_to_process, target, scale_val)

                # 5. Store the processed data for the Model Step
                st.session_state.processed_data = processed_results

                st.success("✅ Preprocessing Complete! Your data is now model-ready.")

                # Show a small preview of the shape
                st.info(f"Training set size: {processed_results[0].shape}")

                # Auto-move to next step after a short delay or show 'Next' button
                if st.button("Move to Model Evaluation ➔"):
                    st.session_state.step = 4
                    st.rerun()

            except Exception as e:
                st.error(f"Preprocessing failed: {e}")


elif st.session_state.step == 4:
    st.header("Step 4: Model Evaluation")

    # 1. Retrieve the preprocessed data from Step 3
    # This is the tuple (X_train, X_test, y_train, y_test)
    X_train, X_test, y_train, y_test = st.session_state.processed_data
    scale_val = st.session_state.scale

    st.write("Now, let's see which algorithm performs best on your data.")

    # 2. Ask the user for the Task Type (if not already known)
    task = st.radio("What type of problem are we solving?", ["Classification", "Regression"])
    st.session_state.task = task
    if st.button("🚀 Run Evaluation Race"):
        with st.spinner("Training multiple models to find the winner..."):

            if task == "Classification":
                import model_classification as mc

                results = mc.Evaluate(X_train, X_test, y_train, y_test)
            else:
                import model_regression as mr

                results = mr.Evaluate(X_train, X_test, y_train, y_test)

            # 3. Store results in session state to display them
            st.session_state.eval_results = results

    # 4. Display the Results Table
    if 'eval_results' in st.session_state:
        st.subheader("🏆 Leaderboard")
        st.table(st.session_state.eval_results)

        # 5. Let the user choose the winner for the final step (Tuning)
        model_names = [r['Model'] for r in st.session_state.eval_results]
        selected_model = st.selectbox("Which model would you like to Hyper-tune and Export?", model_names)

        st.session_state.chosen_model_name = selected_model

        if st.button("Proceed to Tuning & Export ➔"):
            st.session_state.step = 5
            st.rerun()

elif st.session_state.step == 5:
    st.header("🔥 Hyperparameter Tuning & Export")


    def clean_params(params):
        cleaned = {}
        for k, v in params.items():
            if isinstance(v, (np.integer, np.int64)):
                cleaned[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                cleaned[k] = float(v)
            else:
                cleaned[k] = v
        return cleaned
    # get data
    X_train, X_test, y_train, y_test = st.session_state.processed_data
    selected_model = st.session_state.chosen_model_name
    task = st.session_state.task

    st.write(f"Selected Model: **{selected_model}**")

    st.warning("⚠️ Tuning may take some time depending on dataset size.")

    if st.button("🚀 Start Hyperparameter Tuning"):

        with st.spinner("Tuning in progress... ⏳"):

            # 🔹 REGRESSION
            if task == "Regression":
                import model_regression as mr

                model_info = mr.models[selected_model]

                best_model, best_params, best_score = mr.tune_model(
                    selected_model,
                    model_info["model"],
                    X_train,
                    y_train,
                    model_info["scale"]
                )

            # 🔹 CLASSIFICATION
            else:
                import model_classification as mc

                model_info = mc.models[selected_model]

                best_model, best_params, best_score = mc.tune_model(
                    selected_model,
                    model_info["model"],
                    X_train,
                    y_train,
                    model_info["scale"]
                )

        # ✅ Show results
        st.success("✅ Tuning Complete!")

        st.subheader("Best Parameters")
        cleaned_params = clean_params(best_params)
        st.json(cleaned_params)

        st.subheader("Best CV Score")
        st.write(best_score)

        # ✅ Save model
        joblib.dump(best_model, "best_model.pkl")

        # ✅ Download button
        with open("best_model.pkl", "rb") as f:
            st.download_button(
                label="📥 Download Tuned Model",
                data=f,
                file_name="best_model.pkl"
            )
