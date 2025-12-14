import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .insight-card {
        background: #f8f9fa;
        border-left: 4px solid #2E86AB;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .recommendation-card {
        background: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-card {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file_path):
    """Load and preprocess the Superstore dataset"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    
    if df is None:
        raise ValueError("Could not read the CSV file with any supported encoding")
    
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    df['DayOfWeek'] = df['Order Date'].dt.dayofweek
    df['WeekOfYear'] = df['Order Date'].dt.isocalendar().week
    return df


@st.cache_data
def prepare_prophet_data(df, date_col='Order Date', target_col='Sales', agg_freq='D'):
    """Prepare data for Prophet model"""
    if agg_freq == 'D':
        daily_sales = df.groupby(date_col)[target_col].sum().reset_index()
    elif agg_freq == 'W':
        df_temp = df.copy()
        df_temp['Week'] = df_temp[date_col].dt.to_period('W').apply(lambda r: r.start_time)
        daily_sales = df_temp.groupby('Week')[target_col].sum().reset_index()
        daily_sales.columns = [date_col, target_col]
    else:
        df_temp = df.copy()
        df_temp['Month'] = df_temp[date_col].dt.to_period('M').apply(lambda r: r.start_time)
        daily_sales = df_temp.groupby('Month')[target_col].sum().reset_index()
        daily_sales.columns = [date_col, target_col]
    
    daily_sales.columns = ['ds', 'y']
    daily_sales = daily_sales.sort_values('ds').reset_index(drop=True)
    return daily_sales


def get_us_holidays():
    """Get US holidays dataframe for Prophet"""
    holidays = pd.DataFrame({
        'holiday': 'us_holiday',
        'ds': pd.to_datetime([
            '2014-01-01', '2014-07-04', '2014-11-27', '2014-12-25',
            '2015-01-01', '2015-07-04', '2015-11-26', '2015-12-25',
            '2016-01-01', '2016-07-04', '2016-11-24', '2016-12-25',
            '2017-01-01', '2017-07-04', '2017-11-23', '2017-12-25',
            '2018-01-01', '2018-07-04', '2018-11-22', '2018-12-25',
            '2019-01-01', '2019-07-04', '2019-11-28', '2019-12-25',
            '2020-01-01', '2020-07-04', '2020-11-26', '2020-12-25',
            '2021-01-01', '2021-07-04', '2021-11-25', '2021-12-25',
            '2022-01-01', '2022-07-04', '2022-11-24', '2022-12-25',
            '2023-01-01', '2023-07-04', '2023-11-23', '2023-12-25',
            '2024-01-01', '2024-07-04', '2024-11-28', '2024-12-25',
            '2025-01-01', '2025-07-04', '2025-11-27', '2025-12-25',
        ]),
        'lower_window': 0,
        'upper_window': 1,
    })
    
    black_friday = pd.DataFrame({
        'holiday': 'black_friday',
        'ds': pd.to_datetime([
            '2014-11-28', '2015-11-27', '2016-11-25', '2017-11-24',
            '2018-11-23', '2019-11-29', '2020-11-27', '2021-11-26',
            '2022-11-25', '2023-11-24', '2024-11-29', '2025-11-28',
        ]),
        'lower_window': 0,
        'upper_window': 3,
    })
    
    return pd.concat([holidays, black_friday])


@st.cache_resource
def train_prophet_model(prophet_df, forecast_periods, freq='D', include_holidays=True, 
                        seasonality_mode='multiplicative', changepoint_prior=0.05,
                        yearly_seasonality=True, weekly_seasonality=True):
    """Train Prophet model and generate forecast"""
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality if freq == 'D' else False,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior,
        holidays=get_us_holidays() if include_holidays else None
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_periods, freq=freq)
    forecast = model.predict(future)
    return model, forecast


def create_xgboost_features(df):
    """Create features for XGBoost model"""
    features = pd.DataFrame()
    features['ds'] = df['ds']
    features['y'] = df['y']
    features['dayofweek'] = df['ds'].dt.dayofweek
    features['month'] = df['ds'].dt.month
    features['quarter'] = df['ds'].dt.quarter
    features['year'] = df['ds'].dt.year
    features['dayofyear'] = df['ds'].dt.dayofyear
    features['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    
    for lag in [1, 7, 14, 30]:
        if len(df) > lag:
            features[f'lag_{lag}'] = df['y'].shift(lag)
    
    features['rolling_mean_7'] = df['y'].rolling(window=7, min_periods=1).mean()
    features['rolling_std_7'] = df['y'].rolling(window=7, min_periods=1).std()
    features['rolling_mean_30'] = df['y'].rolling(window=30, min_periods=1).mean()
    
    return features.fillna(0)


@st.cache_resource
def train_xgboost_model(prophet_df, forecast_periods, freq='D'):
    """Train XGBoost model and generate forecast"""
    features = create_xgboost_features(prophet_df)
    
    feature_cols = [col for col in features.columns if col not in ['ds', 'y']]
    X = features[feature_cols].values
    y = features['y'].values
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    
    last_date = prophet_df['ds'].max()
    if freq == 'D':
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods, freq='D')
    elif freq == 'W':
        future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=forecast_periods, freq='W')
    else:
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
    
    all_dates = pd.concat([prophet_df[['ds', 'y']], 
                           pd.DataFrame({'ds': future_dates, 'y': np.nan})]).reset_index(drop=True)
    
    predictions = []
    current_data = prophet_df.copy()
    
    for future_date in future_dates:
        temp_df = pd.concat([current_data, pd.DataFrame({'ds': [future_date], 'y': [np.nan]})]).reset_index(drop=True)
        temp_features = create_xgboost_features(temp_df)
        X_pred = temp_features[feature_cols].iloc[-1:].values
        pred = model.predict(X_pred)[0]
        predictions.append(pred)
        current_data = pd.concat([current_data, pd.DataFrame({'ds': [future_date], 'y': [pred]})]).reset_index(drop=True)
    
    historical_preds = model.predict(X)
    
    forecast = pd.DataFrame({
        'ds': pd.concat([prophet_df['ds'], pd.Series(future_dates)]).reset_index(drop=True),
        'yhat': np.concatenate([historical_preds, predictions]),
    })
    
    residuals = y - historical_preds
    std_dev = np.std(residuals) if len(residuals) > 0 else 0
    forecast['yhat_lower'] = forecast['yhat'] - 1.96 * std_dev
    forecast['yhat_upper'] = forecast['yhat'] + 1.96 * std_dev
    
    return model, forecast


def calculate_metrics(actual, predicted):
    """Calculate forecast accuracy metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    r2 = r2_score(actual, predicted)
    return mae, rmse, mape, r2


def generate_business_insights(df, forecast, prophet_df, sales_col='Sales'):
    """Generate automated business insights based on data analysis"""
    insights = []
    recommendations = []
    warnings = []
    
    future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]
    if len(future_forecast) > 0:
        first_pred = future_forecast['yhat'].iloc[0]
        last_pred = future_forecast['yhat'].iloc[-1]
        trend_pct = ((last_pred - first_pred) / first_pred) * 100 if first_pred > 0 else 0
        
        if trend_pct > 10:
            insights.append(f"Strong upward trend detected: Sales are projected to increase by {trend_pct:.1f}% over the forecast period.")
            recommendations.append("Consider increasing inventory levels to meet expected demand growth.")
        elif trend_pct < -10:
            warnings.append(f"Declining trend detected: Sales are projected to decrease by {abs(trend_pct):.1f}% over the forecast period.")
            recommendations.append("Review pricing strategy and marketing efforts to counter the declining trend.")
        else:
            insights.append(f"Stable sales trend: Expected variation of {trend_pct:.1f}% over the forecast period.")
    
    if 'Month' in df.columns:
        monthly_sales = df.groupby('Month')[sales_col].sum()
        peak_month = monthly_sales.idxmax()
        low_month = monthly_sales.idxmin()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        insights.append(f"Peak sales month: {month_names[peak_month-1]} (${monthly_sales[peak_month]:,.0f})")
        insights.append(f"Lowest sales month: {month_names[low_month-1]} (${monthly_sales[low_month]:,.0f})")
        
        if peak_month in [11, 12]:
            recommendations.append("Holiday season drives peak sales. Plan promotional campaigns and ensure adequate staffing for Q4.")
    
    if 'Category' in df.columns:
        category_sales = df.groupby('Category')[sales_col].sum()
        top_category = category_sales.idxmax()
        top_pct = (category_sales[top_category] / category_sales.sum()) * 100
        insights.append(f"Top performing category: {top_category} ({top_pct:.1f}% of total sales)")
        
        if top_pct > 50:
            warnings.append(f"High concentration risk: {top_category} accounts for over 50% of sales. Consider diversifying product mix.")
    
    if 'Region' in df.columns:
        region_sales = df.groupby('Region')[sales_col].sum()
        top_region = region_sales.idxmax()
        low_region = region_sales.idxmin()
        gap_pct = ((region_sales[top_region] - region_sales[low_region]) / region_sales[top_region]) * 100
        
        insights.append(f"Regional leader: {top_region} with ${region_sales[top_region]:,.0f} in sales")
        if gap_pct > 40:
            recommendations.append(f"Significant regional gap detected. Consider targeted marketing in {low_region} region to boost sales.")
    
    avg_transaction = df[sales_col].mean()
    if avg_transaction < 100:
        recommendations.append("Average transaction value is relatively low. Consider upselling strategies or bundle offers.")
    
    return insights, recommendations, warnings


def create_forecast_plot(prophet_df, forecast, xgb_forecast=None, title="Sales Forecast"):
    """Create interactive forecast plot with optional XGBoost comparison"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prophet_df['ds'],
        y=prophet_df['y'],
        mode='lines',
        name='Actual Sales',
        line=dict(color='#2E86AB', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Prophet Forecast',
        line=dict(color='#E94F37', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(233,79,55,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Prophet CI',
        showlegend=True
    ))
    
    if xgb_forecast is not None:
        fig.add_trace(go.Scatter(
            x=xgb_forecast['ds'],
            y=xgb_forecast['yhat'],
            mode='lines',
            name='XGBoost Forecast',
            line=dict(color='#4CAF50', width=2, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=xgb_forecast['ds'].tolist() + xgb_forecast['ds'].tolist()[::-1],
            y=xgb_forecast['yhat_upper'].tolist() + xgb_forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(76,175,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='XGBoost CI',
            showlegend=True
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500
    )
    
    return fig


def create_components_plot(model, forecast):
    """Create seasonality components plot"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Overall Trend', 'Yearly Seasonality', 'Weekly Pattern', 'Forecast Components'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', 
                   name='Trend', line=dict(color='#2E86AB')),
        row=1, col=1
    )
    
    if 'yearly' in forecast.columns:
        yearly_data = forecast[['ds', 'yearly']].copy()
        yearly_data['dayofyear'] = yearly_data['ds'].dt.dayofyear
        yearly_avg = yearly_data.groupby('dayofyear')['yearly'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=yearly_avg['dayofyear'], y=yearly_avg['yearly'], mode='lines',
                       name='Yearly', line=dict(color='#A23B72')),
            row=1, col=2
        )
    
    if 'weekly' in forecast.columns:
        weekly_data = forecast[['ds', 'weekly']].copy()
        weekly_data['dayofweek'] = weekly_data['ds'].dt.dayofweek
        weekly_avg = weekly_data.groupby('dayofweek')['weekly'].mean().reset_index()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig.add_trace(
            go.Bar(x=days, y=weekly_avg['weekly'], name='Weekly', marker_color='#F18F01'),
            row=2, col=1
        )
    
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines',
                   name='Predicted', line=dict(color='#C73E1D')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, template='plotly_white')
    return fig


def create_holiday_analysis_plot(model, forecast):
    """Create holiday effect analysis plot"""
    if 'holidays' not in forecast.columns or forecast['holidays'].isna().all():
        return None
    
    holiday_effect = forecast[['ds', 'holidays']].copy()
    holiday_effect = holiday_effect[holiday_effect['holidays'] != 0]
    
    if len(holiday_effect) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=holiday_effect['ds'],
        y=holiday_effect['holidays'],
        marker_color=np.where(holiday_effect['holidays'] > 0, '#4CAF50', '#f44336'),
        name='Holiday Effect'
    ))
    
    fig.update_layout(
        title='Holiday Impact on Sales',
        xaxis_title='Date',
        yaxis_title='Holiday Effect ($)',
        template='plotly_white',
        height=400
    )
    
    return fig


def main():
    st.title("📈 AI-Powered Sales Forecasting Dashboard")
    st.markdown("*Predict future sales trends using advanced time series analysis with Prophet & XGBoost*")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        data_source = st.radio(
            "Data Source",
            ["Use Sample Superstore Data", "Upload Custom CSV"]
        )
        
        if data_source == "Upload Custom CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                date_col = st.selectbox("Select Date Column", df.columns)
                sales_col = st.selectbox("Select Sales Column", df.columns)
                
                try:
                    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
                except Exception as e:
                    st.error(f"Could not parse date column '{date_col}'. Please ensure it contains valid dates.")
                    return
                
                try:
                    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
                    df = df.dropna(subset=[sales_col])
                except Exception as e:
                    st.error(f"Could not parse sales column '{sales_col}'. Please ensure it contains numeric values.")
                    return
                
                df['Year'] = df[date_col].dt.year
                df['Month'] = df[date_col].dt.month
                df['Quarter'] = df[date_col].dt.quarter
            else:
                st.info("Please upload a CSV file")
                return
        else:
            df = load_data("attached_assets/Sample_-_Superstore_1765377709782.csv")
            date_col = 'Order Date'
            sales_col = 'Sales'
        
        st.divider()
        st.subheader("📅 Forecast Settings")
        
        agg_frequency = st.selectbox(
            "Aggregation Frequency",
            ["Daily", "Weekly", "Monthly"],
            index=1
        )
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        freq = freq_map[agg_frequency]
        
        forecast_periods = st.slider(
            "Forecast Horizon (periods)",
            min_value=7, max_value=365, value=90,
            help="Number of future periods to forecast"
        )
        
        st.divider()
        st.subheader("🤖 Model Selection")
        
        use_xgboost = st.checkbox("Enable XGBoost Comparison", value=True, 
                                   help="Compare Prophet with XGBoost model")
        
        st.divider()
        st.subheader("🎛️ Scenario Planning")
        
        include_holidays = st.checkbox("Include Holiday Effects", value=True)
        
        seasonality_mode = st.selectbox(
            "Seasonality Mode",
            ["multiplicative", "additive"],
            help="Multiplicative works better when seasonal effects scale with trend"
        )
        
        changepoint_prior = st.slider(
            "Trend Flexibility",
            min_value=0.001, max_value=0.5, value=0.05,
            help="Higher values allow more flexible trend changes"
        )
        
        st.divider()
        st.subheader("🔍 Filters")
        
        if 'Category' in df.columns:
            categories = ['All'] + list(df['Category'].unique())
            selected_category = st.selectbox("Category", categories)
            if selected_category != 'All':
                df = df[df['Category'] == selected_category]
        
        if 'Region' in df.columns:
            regions = ['All'] + list(df['Region'].unique())
            selected_region = st.selectbox("Region", regions)
            if selected_region != 'All':
                df = df[df['Region'] == selected_region]
        
        if 'Segment' in df.columns:
            segments = ['All'] + list(df['Segment'].unique())
            selected_segment = st.selectbox("Segment", segments)
            if selected_segment != 'All':
                df = df[df['Segment'] == selected_segment]
    
    prophet_df = prepare_prophet_data(df, date_col, sales_col, freq)
    
    if len(prophet_df) < 10:
        st.warning("Insufficient data for forecasting. Please adjust your filters to include more data points.")
        return
    
    test_size = max(int(len(prophet_df) * 0.2), 5)
    train_df = prophet_df.iloc[:-test_size].copy()
    test_df = prophet_df.iloc[-test_size:].copy()
    
    with st.spinner("Training Prophet model..."):
        model, forecast = train_prophet_model(
            train_df, forecast_periods + test_size, freq,
            include_holidays=include_holidays,
            seasonality_mode=seasonality_mode,
            changepoint_prior=changepoint_prior
        )
    
    xgb_model, xgb_forecast = None, None
    xgb_metrics = None
    if use_xgboost:
        with st.spinner("Training XGBoost model..."):
            xgb_model, xgb_forecast = train_xgboost_model(train_df, forecast_periods + test_size, freq)
            
            xgb_test_forecast = xgb_forecast[xgb_forecast['ds'].isin(test_df['ds'])]
            xgb_merged_test = test_df.merge(xgb_test_forecast[['ds', 'yhat']], on='ds', how='inner')
            
            if len(xgb_merged_test) > 0 and xgb_merged_test['y'].sum() > 0:
                xgb_metrics = calculate_metrics(xgb_merged_test['y'], xgb_merged_test['yhat'])
    
    test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
    merged_test = test_df.merge(test_forecast[['ds', 'yhat']], on='ds', how='inner')
    
    if len(merged_test) > 0 and merged_test['y'].sum() > 0:
        mae, rmse, mape, r2 = calculate_metrics(merged_test['y'], merged_test['yhat'])
    else:
        mae, rmse, mape, r2 = 0, 0, 0, 0
    
    insights, recommendations, warnings_list = generate_business_insights(df, forecast, prophet_df, sales_col)
    
    st.header("📊 Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = df[sales_col].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")
    
    with col2:
        avg_sales = df[sales_col].mean()
        st.metric("Average Transaction", f"${avg_sales:,.2f}")
    
    with col3:
        st.metric("Prophet R²", f"{r2:.2%}")
    
    with col4:
        forecast_total = forecast[forecast['ds'] > prophet_df['ds'].max()]['yhat'].sum()
        st.metric("Forecasted Sales", f"${forecast_total:,.0f}")
    
    if use_xgboost and xgb_metrics:
        st.subheader("📈 Model Comparison")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prophet MAE", f"${mae:,.2f}")
        with col2:
            st.metric("XGBoost MAE", f"${xgb_metrics[0]:,.2f}")
        with col3:
            st.metric("Prophet R²", f"{r2:.4f}")
        with col4:
            st.metric("XGBoost R²", f"{xgb_metrics[3]:.4f}")
        
        better_model = "Prophet" if r2 > xgb_metrics[3] else "XGBoost"
        st.info(f"🏆 **{better_model}** shows better performance on holdout data based on R² score.")
    
    st.divider()
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Forecast", "🔄 Seasonality", "🎄 Holiday Effects", "💡 Insights", 
        "📊 Trend Analysis", "📋 Data Export", "📉 Model Performance"
    ])
    
    with tab1:
        st.subheader("Sales Forecast - Actual vs Predicted")
        fig_forecast = create_forecast_plot(prophet_df, forecast, xgb_forecast if use_xgboost else None)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.subheader("Forecast Summary")
        future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()].copy()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            next_period_sales = future_forecast.head(7)['yhat'].sum() if len(future_forecast) >= 7 else future_forecast['yhat'].sum()
            st.metric("Next 7 Periods", f"${next_period_sales:,.0f}")
        with col2:
            next_month_sales = future_forecast.head(30)['yhat'].sum() if len(future_forecast) >= 30 else future_forecast['yhat'].sum()
            st.metric("Next 30 Periods", f"${next_month_sales:,.0f}")
        with col3:
            trend_direction = "📈 Upward" if future_forecast['yhat'].iloc[-1] > future_forecast['yhat'].iloc[0] else "📉 Downward"
            st.metric("Trend Direction", trend_direction)
    
    with tab2:
        st.subheader("Seasonality Components")
        fig_components = create_components_plot(model, forecast)
        st.plotly_chart(fig_components, use_container_width=True)
        
        st.markdown("""
        ### Understanding Seasonality
        - **Trend**: Long-term direction of sales over time
        - **Yearly Seasonality**: How sales vary throughout the year (holidays, seasons)
        - **Weekly Pattern**: Sales patterns across days of the week
        """)
    
    with tab3:
        st.subheader("Holiday Effect Analysis")
        if include_holidays:
            holiday_fig = create_holiday_analysis_plot(model, forecast)
            if holiday_fig:
                st.plotly_chart(holiday_fig, use_container_width=True)
            else:
                st.info("No significant holiday effects detected in the forecast period.")
            
            st.markdown("""
            ### Holiday Impact Interpretation
            - **Positive values**: Sales increase during these holidays
            - **Negative values**: Sales decrease during these holidays
            - **Major holidays tracked**: New Year, July 4th, Thanksgiving, Christmas, Black Friday
            """)
        else:
            st.info("Enable 'Include Holiday Effects' in the sidebar to see holiday impact analysis.")
    
    with tab4:
        st.subheader("💡 Automated Business Insights")
        
        if insights:
            st.markdown("### Key Insights")
            for insight in insights:
                st.markdown(f'<div class="insight-card">📊 {insight}</div>', unsafe_allow_html=True)
        
        if recommendations:
            st.markdown("### Recommendations")
            for rec in recommendations:
                st.markdown(f'<div class="recommendation-card">✅ {rec}</div>', unsafe_allow_html=True)
        
        if warnings_list:
            st.markdown("### Warnings")
            for warning in warnings_list:
                st.markdown(f'<div class="warning-card">⚠️ {warning}</div>', unsafe_allow_html=True)
    
    with tab5:
        st.subheader("Sales Trend Analysis")
        
        if 'Category' in df.columns:
            category_sales = df.groupby('Category')[sales_col].sum().reset_index()
            fig_cat = px.pie(category_sales, values=sales_col, names='Category',
                            title='Sales by Category', hole=0.4)
            st.plotly_chart(fig_cat, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Region' in df.columns:
                region_sales = df.groupby('Region')[sales_col].sum().reset_index()
                fig_region = px.bar(region_sales, x='Region', y=sales_col,
                                   title='Sales by Region', color='Region')
                st.plotly_chart(fig_region, use_container_width=True)
        
        with col2:
            if 'Year' in df.columns:
                yearly_sales = df.groupby('Year')[sales_col].sum().reset_index()
                fig_yearly = px.line(yearly_sales, x='Year', y=sales_col,
                                    title='Yearly Sales Trend', markers=True)
                st.plotly_chart(fig_yearly, use_container_width=True)
        
        if 'Sub-Category' in df.columns:
            st.subheader("Top & Bottom Performing Products")
            subcat_sales = df.groupby('Sub-Category')[sales_col].sum().reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                top_5 = subcat_sales.nlargest(5, sales_col)
                fig_top = px.bar(top_5, x=sales_col, y='Sub-Category', orientation='h',
                                title='Top 5 Sub-Categories', color=sales_col,
                                color_continuous_scale='Greens')
                st.plotly_chart(fig_top, use_container_width=True)
            
            with col2:
                bottom_5 = subcat_sales.nsmallest(5, sales_col)
                fig_bottom = px.bar(bottom_5, x=sales_col, y='Sub-Category', orientation='h',
                                   title='Bottom 5 Sub-Categories', color=sales_col,
                                   color_continuous_scale='Reds')
                st.plotly_chart(fig_bottom, use_container_width=True)
    
    with tab6:
        st.subheader("📥 Export Data for Power BI")
        st.markdown("""
        Download the processed data and forecasts to import into Power BI for your final dashboard.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Forecast Data")
            forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].copy()
            forecast_export.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound', 'Trend']
            forecast_export['Actual'] = forecast_export['Date'].map(
                dict(zip(prophet_df['ds'], prophet_df['y']))
            )
            forecast_export['Model'] = 'Prophet'
            
            if use_xgboost and xgb_forecast is not None:
                xgb_export = xgb_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                xgb_export.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
                xgb_export['Trend'] = np.nan
                xgb_export['Actual'] = xgb_export['Date'].map(
                    dict(zip(prophet_df['ds'], prophet_df['y']))
                )
                xgb_export['Model'] = 'XGBoost'
                forecast_export = pd.concat([forecast_export, xgb_export])
            
            csv_forecast = forecast_export.to_csv(index=False)
            st.download_button(
                label="📊 Download Forecast CSV",
                data=csv_forecast,
                file_name="sales_forecast.csv",
                mime="text/csv"
            )
            
            buffer = io.BytesIO()
            forecast_export.to_excel(buffer, index=False, engine='openpyxl')
            st.download_button(
                label="📑 Download Forecast Excel",
                data=buffer.getvalue(),
                file_name="sales_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            st.markdown("### Original Data with Features")
            export_df = df.copy()
            csv_original = export_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Original Data CSV",
                data=csv_original,
                file_name="superstore_with_features.csv",
                mime="text/csv"
            )
            
            buffer2 = io.BytesIO()
            export_df.to_excel(buffer2, index=False, engine='openpyxl')
            st.download_button(
                label="📑 Download Original Data Excel",
                data=buffer2.getvalue(),
                file_name="superstore_with_features.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.divider()
        st.markdown("### API Endpoint for Power BI")
        st.markdown("""
        For live data integration with Power BI, use the JSON export below:
        """)
        
        api_data = {
            "forecast_summary": {
                "total_forecasted_sales": float(forecast_total),
                "forecast_periods": forecast_periods,
                "model": "Prophet + XGBoost" if use_xgboost else "Prophet",
                "metrics": {
                    "prophet_r2": float(r2),
                    "prophet_mae": float(mae),
                    "xgboost_r2": float(xgb_metrics[3]) if xgb_metrics else None,
                    "xgboost_mae": float(xgb_metrics[0]) if xgb_metrics else None,
                }
            },
            "forecast_data": forecast_export.head(100).to_dict(orient='records')
        }
        
        json_export = json.dumps(api_data, indent=2, default=str)
        st.download_button(
            label="📄 Download JSON for API",
            data=json_export,
            file_name="forecast_api_data.json",
            mime="application/json"
        )
        
        st.divider()
        st.markdown("""
        ### Power BI Import Instructions
        1. Download the Excel or CSV files above
        2. Open Power BI Desktop
        3. Click **Get Data** → **Excel** or **Text/CSV**
        4. Select the downloaded file
        5. Use the Date column for your time axis
        6. Create visualizations comparing Actual vs Forecast
        7. For live integration, use **Get Data** → **JSON** with the API export
        """)
    
    with tab7:
        st.subheader("Model Performance Metrics (Holdout Evaluation)")
        st.info(f"Metrics calculated on {test_size} holdout periods (20% of data) for unbiased accuracy assessment.")
        
        st.markdown("### Prophet Model")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"${mae:,.2f}", help="Mean Absolute Error on holdout data")
        with col2:
            st.metric("RMSE", f"${rmse:,.2f}", help="Root Mean Square Error on holdout data")
        with col3:
            st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error on holdout data")
        with col4:
            st.metric("R² Score", f"{r2:.4f}", help="Coefficient of Determination on holdout data")
        
        if use_xgboost and xgb_metrics:
            st.markdown("### XGBoost Model")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"${xgb_metrics[0]:,.2f}")
            with col2:
                st.metric("RMSE", f"${xgb_metrics[1]:,.2f}")
            with col3:
                st.metric("MAPE", f"{xgb_metrics[2]:.2f}%")
            with col4:
                st.metric("R² Score", f"{xgb_metrics[3]:.4f}")
        
        st.subheader("Actual vs Predicted Comparison (Holdout Data)")
        fig_comparison = go.Figure()
        if len(merged_test) > 0:
            fig_comparison.add_trace(go.Scatter(
                x=merged_test['y'], y=merged_test['yhat'],
                mode='markers', name='Prophet',
                marker=dict(color='#E94F37', size=10, opacity=0.7)
            ))
            
            if use_xgboost and xgb_metrics:
                xgb_test_forecast = xgb_forecast[xgb_forecast['ds'].isin(test_df['ds'])]
                xgb_merged = test_df.merge(xgb_test_forecast[['ds', 'yhat']], on='ds', how='inner')
                fig_comparison.add_trace(go.Scatter(
                    x=xgb_merged['y'], y=xgb_merged['yhat'],
                    mode='markers', name='XGBoost',
                    marker=dict(color='#4CAF50', size=10, opacity=0.7)
                ))
            
            max_val = max(merged_test['y'].max(), merged_test['yhat'].max())
            fig_comparison.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(color='gray', dash='dash')
            ))
        
        fig_comparison.update_layout(
            title='Actual vs Predicted Sales (Holdout Test Set)',
            xaxis_title='Actual Sales ($)',
            yaxis_title='Predicted Sales ($)',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.subheader("Residual Analysis")
        if len(merged_test) > 0:
            residuals = merged_test['y'] - merged_test['yhat']
            fig_residuals = px.histogram(residuals, nbins=50, 
                                         title='Distribution of Prediction Errors (Holdout Data)',
                                         labels={'value': 'Residual ($)', 'count': 'Frequency'})
            fig_residuals.update_layout(template='plotly_white')
            st.plotly_chart(fig_residuals, use_container_width=True)
        else:
            st.warning("No holdout data available for residual analysis.")
    
    st.divider()
    st.markdown("""
    ---
    **AI Sales Forecasting Dashboard** | Built with Prophet, XGBoost & Streamlit  
    *Data-driven insights for smarter business decisions*
    """)


if __name__ == "__main__":
    main()
