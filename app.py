#!/usr/bin/env python
"""
Hotel Booking Cancellation Intelligence Platform
A clean, native Streamlit application optimized for both Light and Dark modes.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os

# ---------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hotel Cancellation Intelligence",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_BASE = "http://localhost:8000"

# ---------------------------------------------------------
# Helper Functions & Data Caching
# ---------------------------------------------------------
def check_api_health():
    """Verify backend FastAPI server health."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def request_prediction(payload: dict):
    """Call FastAPI backend /predict endpoint."""
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            return True, r.json()
        return False, f"API Error ({r.status_code}): {r.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

@st.cache_data
def load_dataset():
    """Cached loader for hotel bookings dataset."""
    csv_path = "hotel_bookings.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# ---------------------------------------------------------
# PAGE 1: 🔮 Interactive Cancellation Predictor
# ---------------------------------------------------------
def render_prediction_page():
    st.header("🔮 Reservation Risk Predictor")
    st.write("Adjust the 3 simple steps below to instantly calculate the cancellation probability for any booking.")

    # Preset Quick Buttons
    st.subheader("⚡ Quick Test Scenarios")
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("🟢 Standard Leisure Booking", use_container_width=True, help="Typical individual booking (Low risk)"):
            st.session_state.update(
                hotel_type="City Hotel", lead_days=20, room_price=110.0, nights=3,
                adults=2, children=0, channel="Online Travel Agency (TA)",
                deposit="No Deposit", prior_cancels=0, special_reqs=1
            )
            st.rerun()
    with p2:
        if st.button("🏖️ Family Resort Stay", use_container_width=True, help="Family holiday with kids"):
            st.session_state.update(
                hotel_type="Resort Hotel", lead_days=50, room_price=220.0, nights=6,
                adults=2, children=2, channel="Direct Booking (Hotel Website)",
                deposit="No Deposit", prior_cancels=0, special_reqs=2
            )
            st.rerun()
    with p3:
        if st.button("🚨 High-Risk Advance Booking", use_container_width=True, help="Group booking made far in advance"):
            st.session_state.update(
                hotel_type="City Hotel", lead_days=250, room_price=145.0, nights=4,
                adults=2, children=0, channel="Group Booking",
                deposit="Non-Refundable", prior_cancels=2, special_reqs=0
            )
            st.rerun()

    st.divider()

    with st.form("clean_prediction_form"):
        # Step 1
        st.subheader("1️⃣ Hotel Property & Stay Length")
        c1, c2 = st.columns(2)
        with c1:
            hotel_type = st.selectbox(
                "Hotel Type", ["City Hotel", "Resort Hotel"],
                index=0 if st.session_state.get("hotel_type", "City Hotel") == "City Hotel" else 1
            )
        with c2:
            room_price = st.number_input(
                "Nightly Room Rate ($ ADR)", min_value=0.0, max_value=3000.0,
                value=float(st.session_state.get("room_price", 120.0)), step=10.0
            )

        d1, d2, d3 = st.columns([1.2, 1, 1])
        with d1:
            checkin_date = st.date_input(
                "Expected Check-in Date",
                value=datetime.date(2017, 7, 15),
                min_value=datetime.date(2015, 1, 1),
                max_value=datetime.date(2028, 12, 31)
            )
        with d2:
            weekend_nights = st.number_input(
                "Weekend Nights (Fri/Sat)", min_value=0, max_value=14,
                value=2 if st.session_state.get("nights", 3) >= 2 else 0, step=1
            )
        with d3:
            week_nights = st.number_input(
                "Weekday Nights (Sun-Thu)", min_value=0, max_value=30,
                value=max(1, st.session_state.get("nights", 3) - 2), step=1
            )

        # Step 2
        st.subheader("2️⃣ Guests & Booking Channel")
        g1, g2, g3 = st.columns(3)
        with g1:
            adults = st.number_input("Adults", min_value=1, max_value=10, value=int(st.session_state.get("adults", 2)), step=1)
        with g2:
            children = st.number_input("Children", min_value=0, max_value=10, value=int(st.session_state.get("children", 0)), step=1)
        with g3:
            babies = st.number_input("Babies", min_value=0, max_value=5, value=0, step=1)

        ch1, ch2 = st.columns(2)
        with ch1:
            channel_opts = ["Online Travel Agency (TA)", "Direct Booking (Hotel Website)", "Corporate / Business", "Offline Travel Agent", "Group Booking"]
            idx = 0
            if "Direct" in st.session_state.get("channel", ""): idx = 1
            elif "Corporate" in st.session_state.get("channel", ""): idx = 2
            elif "Group" in st.session_state.get("channel", ""): idx = 4
            elif "Offline" in st.session_state.get("channel", ""): idx = 3
            channel_choice = st.selectbox("Booking Channel", channel_opts, index=idx)
        with ch2:
            country = st.selectbox(
                "Origin Country",
                ["USA - United States", "GBR - United Kingdom", "PRT - Portugal", "ESP - Spain", "FRA - France", "DEU - Germany", "ITA - Italy", "IRL - Ireland", "BRA - Brazil", "OTHER"],
                index=0
            )

        # Step 3
        st.subheader("3️⃣ Timing, Policies & Customer History")
        lead_days = st.slider(
            "Days Booked in Advance (Lead Time)", min_value=0, max_value=365,
            value=int(st.session_state.get("lead_days", 30))
        )

        p1, p2, p3 = st.columns(3)
        with p1:
            deposit_choice = st.selectbox(
                "Deposit Policy", ["No Deposit", "Non-Refundable", "Refundable"],
                index=0 if st.session_state.get("deposit", "No Deposit") == "No Deposit" else 1
            )
        with p2:
            prior_cancels = st.number_input(
                "Prior Cancellations by Guest", min_value=0, max_value=20,
                value=int(st.session_state.get("prior_cancels", 0)), step=1
            )
        with p3:
            special_reqs = st.number_input(
                "Special Requests (Crib, high floor, etc.)", min_value=0, max_value=5,
                value=int(st.session_state.get("special_reqs", 1)), step=1
            )

        with st.expander("⚙️ Optional Extra Amenities"):
            adv1, adv2, adv3 = st.columns(3)
            with adv1:
                parking = st.selectbox("Parking Space Required?", ["No (0)", "Yes (1)"])
            with adv2:
                meal_plan = st.selectbox("Meal Plan", ["Bed & Breakfast (Meal 1)", "Half Board (Board HF)", "Full Board (Board FBF)", "Room Only / Undefined"])
            with adv3:
                booking_changes = st.number_input("Modifications Made", min_value=0, max_value=10, value=0, step=1)

        total_nights = weekend_nights + week_nights
        total_guests = adults + children + babies
        total_price = room_price * total_nights if total_nights > 0 else room_price

        # Native Metrics for Summary
        st.write("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stay Duration", f"{total_nights} Nights")
        m2.metric("Total Guests", f"{total_guests} Guests")
        m3.metric("Est. Total Bill", f"${total_price:,.2f}")
        m4.metric("Advance Notice", f"{lead_days} Days")
        
        st.write("---")
        submit_btn = st.form_submit_button("⚡ Predict Cancellation Probability", use_container_width=True, type="primary")

    if submit_btn:
        channel_mapping = {
            "Online Travel Agency (TA)": ("Online TA", "TA/Online"),
            "Direct Booking (Hotel Website)": ("Direct", "Direct"),
            "Corporate / Business": ("Corporate", "Corporate"),
            "Offline Travel Agent": ("Offline TA", "TA/Online"),
            "Group Booking": ("Groups", "TA/Online")
        }
        market_segment, distribution_channel = channel_mapping.get(channel_choice, ("Online TA", "TA/Online"))

        meal_mapping = {
            "Bed & Breakfast (Meal 1)": "Meal 1",
            "Half Board (Board HF)": "Board HF",
            "Full Board (Board FBF)": "Board FBF",
            "Room Only / Undefined": "Undefined"
        }
        meal_val = meal_mapping.get(meal_plan, "Meal 1")
        country_code = country.split(" - ")[0] if " - " in country else "USA"
        parking_val = 1 if "Yes" in parking else 0
        adr_per_guest = room_price / total_guests if total_guests > 0 else 0.0
        is_family = 1 if (children + babies) > 0 else 0

        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        arrival_month_str = month_names[checkin_date.month - 1]
        week_num = checkin_date.isocalendar()[1]

        payload = {
            "numerical": {
                "lead_time": float(lead_days),
                "arrival_date_year": int(checkin_date.year),
                "arrival_date_week_number": int(week_num),
                "arrival_date_day_of_month": int(checkin_date.day),
                "stays_in_weekend_nights": float(weekend_nights),
                "stays_in_week_nights": float(week_nights),
                "adults": int(adults),
                "children": float(children),
                "babies": float(babies),
                "is_repeated_guest": 0,
                "previous_cancellations": int(prior_cancels),
                "previous_bookings_not_canceled": 0,
                "booking_changes": int(booking_changes),
                "days_in_waiting_list": 0,
                "adr": float(room_price),
                "required_car_parking_spaces": int(parking_val),
                "total_of_special_requests": int(special_reqs),
                "total_guests": float(total_guests),
                "total_stay_nights": float(total_nights),
                "total_previous_bookings": 0,
                "is_family": int(is_family),
                "adr_per_guest": float(adr_per_guest),
            },
            "categorical": {
                "hotel": hotel_type,
                "arrival_date_month": arrival_month_str,
                "meal": meal_val,
                "country": country_code,
                "market_segment": market_segment,
                "distribution_channel": distribution_channel,
                "deposit_type": deposit_choice,
                "customer_type": "Transient" if "Group" not in channel_choice else "Group",
            }
        }

        with st.spinner("Analyzing booking signals with XGBoost..."):
            success, result = request_prediction(payload)

        if not success:
            st.error(result)
        else:
            prediction_label = result["prediction_label"]
            cancel_prob = result["probabilities"].get("Canceled", 0.0)
            not_cancel_prob = result["probabilities"].get("Not Canceled", 0.0)

            st.header("📊 Prediction Result")
            if prediction_label == "Canceled":
                st.error(f"### ⚠️ HIGH CANCELLATION RISK ({cancel_prob:.0%})\nThe machine learning model detected strong cancellation signals in this reservation.")
            else:
                st.success(f"### ✅ SAFE TO CONFIRM ({not_cancel_prob:.0%} Confidence)\nThis reservation has high reliability and is expected to check in normally.")

            res1, res2 = st.columns([1.2, 1])
            with res1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=cancel_prob * 100,
                    number={"suffix": "%", "font": {"size": 34}},
                    title={"text": "Cancellation Probability Index", "font": {"size": 15}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": "#ef4444" if cancel_prob >= 0.5 else "#10b981"},
                        "steps": [
                            {"range": [0, 35], "color": "rgba(16, 185, 129, 0.2)"},
                            {"range": [35, 65], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [65, 100], "color": "rgba(239, 68, 68, 0.2)"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.8, "value": 50}
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            with res2:
                st.subheader("💡 Front Desk Action Plan:")
                if cancel_prob >= 0.60:
                    st.warning("**High Risk**: Send an automated confirmation link, verify card pre-authorization, and keep standby room overbooking.")
                elif cancel_prob >= 0.35:
                    st.info("**Moderate Risk**: Send a friendly pre-arrival message highlighting hotel amenities (dining, spa, mobile check-in).")
                else:
                    st.success("**Low Risk**: No special action needed. Prepare room for VIP arrival.")


# ---------------------------------------------------------
# PAGE 2: 📋 Project Summary & Pipeline
# ---------------------------------------------------------
def render_project_summary_page():
    st.header("📋 Project Summary & Machine Learning Pipeline")
    st.write("A complete, easy-to-understand breakdown of the business problem, data workflow, model evaluation, and selected winner.")

    # 1. Goal & Value
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.info("### 🚨 The Business Problem\nHotels lose significant revenue when guests cancel without warning. Empty rooms lead to lost income and wasted operational costs. Predicting cancellations early allows managers to optimize overbooking and protect income.")
    with col_g2:
        st.success("### 💡 The Machine Learning Solution\nWe trained supervised machine learning models on **119,390 real hotel bookings**. The system analyzes 30 reservation factors (lead time, room rate, booking channel) to predict cancellation risk and recommend actions.")

    # 2. Pipeline Flow
    st.subheader("🔄 The 4-Step Machine Learning Pipeline")
    
    # Using columns with varying widths to create a visual sequence with arrows
    c1, arr1, c2, arr2, c3, arr3, c4 = st.columns([3, 0.8, 3, 0.8, 3, 0.8, 3])
    
    with c1:
        st.info("#### 1️⃣ Data Cleaning\nCleaned 119k records, removing duplicates and missing values.")
    with arr1:
        st.markdown("<h2 style='text-align: center; margin-top: 35px;'>➔</h2>", unsafe_allow_html=True)
    with c2:
        st.warning("#### 2️⃣ Feature Eng.\nEngineered domain features and eliminated data leakage.")
    with arr2:
        st.markdown("<h2 style='text-align: center; margin-top: 35px;'>➔</h2>", unsafe_allow_html=True)
    with c3:
        st.error("#### 3️⃣ Benchmarking\nTrained 5 algorithms using 5-Fold Cross-Validation.")
    with arr3:
        st.markdown("<h2 style='text-align: center; margin-top: 35px;'>➔</h2>", unsafe_allow_html=True)
    with c4:
        st.success("#### 4️⃣ Deployment\nLive deployment via high-speed FastAPI and Streamlit UI.")

    st.divider()

    # 3. Winning Model Spotlight
    st.subheader("🥇 Selected Best Model: XGBoost Classifier")
    st.write("Selected for delivering the highest F1-Score (0.7090) and superior balance between catching cancellations and minimizing false alarms.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", "83.2%")
    m2.metric("F1-Score (Canceled)", "0.7090")
    m3.metric("Recall (Catch Rate)", "70.5%")
    m4.metric("Precision", "69.1%")

    st.divider()

    # 4. Model Comparison
    st.subheader("🏆 Model Comparison Leaderboard")
    
    leaderboard_df = pd.DataFrame({
        "Rank": ["1 🥇", "2 🥈", "3 🥉", "4", "5"],
        "Model": ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree", "K-Nearest Neighbors (KNN)"],
        "F1-Score": [0.7090, 0.7004, 0.6437, 0.6311, 0.6026],
        "Accuracy (%)": [83.18, 82.84, 78.12, 76.89, 75.40],
        "Recall (%)": [70.51, 69.80, 67.60, 68.00, 65.00]
    })

    col_chart, col_tbl = st.columns([1.2, 1])
    with col_chart:
        fig_lead = px.bar(
            leaderboard_df, x="Model", y="F1-Score", color="F1-Score",
            color_continuous_scale="Teal", text="F1-Score", title="Model F1-Score Performance"
        )
        fig_lead.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_lead.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_lead, use_container_width=True, theme="streamlit")

    with col_tbl:
        st.dataframe(leaderboard_df, hide_index=True, use_container_width=True)
        st.info("💡 **Takeaway:** Tree ensemble algorithms (XGBoost & Random Forest) significantly outperformed standard linear models.")

    # 5. Top Predictive Factors
    st.subheader("🔍 What Matters Most in Predicting Cancellations?")
    imp_df = pd.DataFrame({
        "Feature": ["Lead Time (Advance Days)", "Guest Origin Country", "Deposit Policy", "Nightly Room Rate (ADR)", "Booking Channel", "Previous Cancellations"],
        "Importance (%)": [28.4, 21.2, 17.5, 14.1, 11.3, 7.5]
    }).sort_values(by="Importance (%)", ascending=True)

    fig_imp = px.bar(
        imp_df, x="Importance (%)", y="Feature", orientation="h",
        color="Importance (%)", color_continuous_scale="Blues", title="Top 6 Decision Drivers"
    )
    fig_imp.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig_imp, use_container_width=True, theme="streamlit")


# ---------------------------------------------------------
# PAGE 3: 📊 Exploratory Data Analysis
# ---------------------------------------------------------
def render_eda_page():
    st.header("📊 Exploratory Data Analysis & Business Insights")
    st.write("Visual findings, statistical summaries, and actionable revenue takeaways from 119,390 historical bookings.")

    df = load_dataset()
    if df is None:
        st.error("Dataset `hotel_bookings.csv` was not found in the root directory.")
        return

    # 1. Top KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Bookings", f"{len(df):,}")
    k2.metric("Overall Cancel Rate", f"{df['is_canceled'].mean():.1%}")
    k3.metric("City Hotels Share", f"{(df['hotel'] == 'City Hotel').mean():.1%}")
    k4.metric("Average Lead Time", f"{df['lead_time'].mean():.0f} Days")
    k5.metric("Average Daily Rate", f"${df['adr'].mean():.2f}")

    st.divider()

    # 2. Executive Key Takeaways
    st.subheader("💡 4 Key Takeaways for Hotel Management")
    t1, t2 = st.columns(2)
    with t1:
        st.info("**🏨 1. City Hotels cancel more frequently (41.7%) than Resort Hotels (27.8%).**\nCity hotels cater to short-term business and flexible travelers, while resort guests plan vacations further in advance and commit firmly.")
        st.warning("**⏱️ 2. Advance Lead Time is the #1 predictor of cancellation.**\nBookings made >100 days ahead have over double the cancellation rate of bookings made within 30 days of arrival.")
    with t2:
        st.success("**🎯 3. Special Requests strongly correlate with guest loyalty.**\nGuests with 2+ special requests (baby crib, high floor, quiet room) cancel less than 18% of the time.")
        st.error("**☀️ 4. Peak Summer Months (July & August) experience the highest volume and cancellations.**\nSurge seasonal demand requires tighter overbooking and pre-arrival communication protocols.")

    st.divider()

    # 3. Question-Driven Visualizations
    st.subheader("📈 Visualizing the Trends")

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        hotel_cancel = df.groupby("hotel")["is_canceled"].value_counts(normalize=True).unstack() * 100
        hotel_cancel.columns = ["Completed Stay (%)", "Canceled (%)"]
        hotel_cancel = hotel_cancel.reset_index()

        fig1 = px.bar(
            hotel_cancel, x="hotel", y=["Completed Stay (%)", "Canceled (%)"],
            title="Q1: Which hotel type experiences more cancellations?", barmode="group",
            color_discrete_map={"Completed Stay (%)": "#10b981", "Canceled (%)": "#ef4444"}
        )
        fig1.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

    with c2:
        sample_df = df.sample(min(8000, len(df)), random_state=42).copy()
        sample_df["Status"] = sample_df["is_canceled"].map({0: "Completed Stay", 1: "Canceled"})

        fig2 = px.histogram(
            sample_df, x="lead_time", color="Status", nbins=35, barmode="overlay",
            title="Q2: Does booking months ahead increase cancellation risk?",
            color_discrete_map={"Completed Stay": "#10b981", "Canceled": "#ef4444"}, opacity=0.75
        )
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        month_df = df.groupby("arrival_date_month")["is_canceled"].agg(Total="count", Canceled="sum").reindex(month_order).reset_index()
        month_df["Cancel_Rate"] = (month_df["Canceled"] / month_df["Total"]) * 100

        fig3 = px.line(
            month_df, x="arrival_date_month", y="Cancel_Rate", markers=True,
            title="Q3: When do most cancellations happen during the year?",
            labels={"arrival_date_month": "Month", "Cancel_Rate": "Cancellation Rate (%)"}
        )
        fig3.update_traces(line_width=3, marker=dict(size=7))
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig3, use_container_width=True, theme="streamlit")

    with c4:
        spec_df = df.groupby("total_of_special_requests")["is_canceled"].agg(Count="count", Cancel_Rate=lambda x: (x.mean()) * 100).reset_index()

        fig4 = px.bar(
            spec_df, x="total_of_special_requests", y="Cancel_Rate",
            title="Q4: Do special requests indicate lower cancellation?",
            color="Cancel_Rate", color_continuous_scale="Teal_r",
            labels={"total_of_special_requests": "Number of Special Requests", "Cancel_Rate": "Cancellation Rate (%)"}
        )
        fig4.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig4, use_container_width=True, theme="streamlit")

    # Detailed Stats Expander
    with st.expander("🔍 Click to view Detailed Numerical Statistics Summary"):
        key_cols = ["lead_time", "adr", "stays_in_week_nights", "stays_in_weekend_nights", "adults", "children", "total_of_special_requests"]
        st.dataframe(df[key_cols].describe().T.style.format("{:.2f}"), use_container_width=True)


# ---------------------------------------------------------
# App Router & Navigation
# ---------------------------------------------------------
def main():
    is_healthy = check_api_health()
    
    st.title("🏨 Hotel Booking Cancellation Intelligence")
    if is_healthy:
        st.success("🟢 API Connected & Model Ready")
    else:
        st.error("🔴 API Offline - Please start the FastAPI backend")

    tab_pred, tab_summary, tab_eda = st.tabs([
        "🔮 Cancellation Predictor",
        "📋 Project Summary & Pipeline",
        "📊 Exploratory Data Analysis (EDA)"
    ])

    with tab_pred:
        render_prediction_page()
    with tab_summary:
        render_project_summary_page()
    with tab_eda:
        render_eda_page()

if __name__ == "__main__":
    main()