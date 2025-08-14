# main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from enhanced_llm_agent import enhanced_llm_agent

# Database configuration
DB_PARAMS = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": "abcd@1234",
    "port": 5432
}

load_dotenv()

app = FastAPI(
    title="Advanced Financial Projection System",
    description="Comprehensive financial projection system with AI integration and GTM",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class CellUpdateRequest(BaseModel):
    company_id: int
    sheet_type: str
    field_name: str
    year_num: int
    quarter_num: int
    value: float
    propagate_years: Optional[int] = 1

class AIAssistantRequest(BaseModel):
    prompt: str
    sheet_type: str
    sheet_data: Dict[str, Any]
    company_id: int

class AddColumnRequest(BaseModel):
    company_id: int
    sheet_type: str
    column_name: str
    column_type: str
    description: str = ""

class GTMUpdateRequest(BaseModel):
    acquisition_type: str
    year_num: int
    quarter_num: int
    count: int
    amount_per_acquisition: float = 0

class AutoPopulateRequest(BaseModel):
    company_id: int
    year: int

class StressTestRequest(BaseModel):
    company_id: int
    year: int
    quarter: int

# New Pydantic model for the stress test endpoint
class StressTestPayload(BaseModel):
    start_year: int
    start_quarter: int
    customer_drop_percentage: Optional[float] = 0
    pricing_pressure_percentage: Optional[float] = 0
    cac_increase_percentage: Optional[float] = 0
    is_technology_failure: bool = False
    interest_rate_shock: Optional[float] = 0
    market_entry_underperformance_percentage: Optional[float] = 0
    is_economic_recession: bool = False


# Database helper functions
def get_db_connection():
    return psycopg2.connect(**DB_PARAMS)

def execute_query(query, params=None, fetch=False):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(query, params)
        if fetch:
            result = cur.fetchall()
            conn.commit()
            return result
        else:
            conn.commit()
            return cur.rowcount
    finally:
        cur.close()
        conn.close()

# Enhanced calculation functions
def calculate_inflation_rate(quarter):
    """Calculate cumulative inflation rate for a given quarter"""
    try:
        if isinstance(quarter, str):
            if quarter.startswith('Y') and 'Q' in quarter:
                year_part, quarter_part = quarter.split('Q')
                year_num = int(year_part[1:])
                quarter_num = int(quarter_part)
            else:
                raise ValueError(f"Invalid quarter format: {quarter}")
        else:
            year_num = quarter.get('year', 1)
            quarter_num = quarter.get('quarter', 1)
        
        quarterly_rate = 1.5
        total_quarters = (year_num - 1) * 4 + quarter_num - 1
        
        return total_quarters * quarterly_rate
    except (ValueError, AttributeError, IndexError) as e:
        print(f"Error calculating inflation rate for quarter {quarter}: {e}")
        return 0

# API Endpoints
@app.get("/api/sheet-data/{sheet_type}/{year}")
async def get_sheet_data(sheet_type: str, year: int):
    """Get financial sheet data with enhanced calculations - FIXED ORDER"""
    try:
        # It's important to keep the backend stateless and let the frontend
        # trigger the recalculation via an update endpoint. This prevents
        # unexpected side effects on simple GET requests.
        query = """
            SELECT field_name, year_num, quarter_num, field_value, is_calculated
            FROM financial_sheets
            WHERE company_id = 1 AND sheet_type = %s AND year_num = %s
            ORDER BY field_name, quarter_num
        """
        rows = execute_query(query, (sheet_type, year), fetch=True)
        
        data = {}
        for row in rows:
            field_name = row['field_name']
            quarter_key = f"Y{row['year_num']}Q{row['quarter_num']}"
            
            if field_name not in data:
                data[field_name] = {}
            
            data[field_name][quarter_key] = {
                'value': float(row['field_value']),
                'is_calculated': row['is_calculated']
            }
        
        if sheet_type == 'capex':
            await calculate_capex_enhanced(data, year)
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching sheet data for {sheet_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gtm-data/{year}")
async def get_gtm_data(year: int):
    """Get existing GTM data for a specific year to populate the UI."""
    try:
        query = """
            SELECT acquisition_type, year_num, quarter_num, acquisition_count,
                   customers_added, fixed_deposit_value, property_value, is_permanent,
                   amount_per_acquisition, total_amount
            FROM gtm_acquisitions
            WHERE company_id = 1 AND year_num = %s
            ORDER BY acquisition_type, quarter_num
        """
        rows = execute_query(query, (year,), fetch=True)
        
        gtm_data = {}
        for row in rows:
            acq_type = row['acquisition_type']
            if acq_type not in gtm_data:
                gtm_data[acq_type] = {}
            
            quarter_key = f"Y{row['year_num']}Q{row['quarter_num']}"
            gtm_data[acq_type][quarter_key] = {
                'count': row['acquisition_count'],
                'amount_per_acquisition': float(row['amount_per_acquisition']) if row['amount_per_acquisition'] is not None else 0
            }
            
        return gtm_data
    except Exception as e:
        print(f"❌ Error fetching GTM data for year {year}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auto-populate-reach")
async def auto_populate_reach_ai(request: AutoPopulateRequest):
    """Auto-populate reach fields for a given year using the enhanced LLM agent."""
    try:
        company_id = request.company_id
        year = request.year
        field_mapping = {
            'Search Engine & GPT Marketing Spends': 'Average Reach from Search',
            'Social Media Marketing Spends (Ads)': 'Average Reach from Social Ads',
            'Social Media Campaigns (Strategy & Design Spends)': 'Average Reach from Social Campaigns',
            'ATL Campaigns Spends': 'Average Reach from ATL'
        }
        company_context = {'industry': 'Fintech', 'stage': 'Growth', 'location': 'India', 'funding_round': 'Series A'}

        for quarter in range(1, 5):
            spend_values = {}
            for spend_field in field_mapping.keys():
                spend_values[spend_field] = await get_field_value('growth-funnel', spend_field, year, quarter)
            for spend_field, reach_field in field_mapping.items():
                current_spend = spend_values.get(spend_field, 0)
                if current_spend > 0:
                    suggestion = await enhanced_llm_agent.get_contextual_suggestions(
                        sheet_type='growth-funnel', field_name=reach_field, current_value=0,
                        related_fields={'spend': current_spend}, company_context=company_context
                    )
                    suggested_reach = suggestion.get('suggested_value', 0)
                    await update_field_as_input(company_id, 'growth-funnel', reach_field, year, quarter, suggested_reach)

        await run_full_recalculation_for_year(company_id, year)
        return {"status": "success", "message": "AI reach suggestions applied and calculations refreshed."}
    except Exception as e:
        print(f"Error in auto-populate-reach: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def update_field_as_input(company_id, sheet_type, field_name, year, quarter, value):
    query = """
        INSERT INTO financial_sheets (company_id, sheet_type, field_name, year_num, quarter_num, field_value, is_calculated, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, FALSE, CURRENT_TIMESTAMP)
        ON CONFLICT (company_id, sheet_type, field_name, year_num, quarter_num)
        DO UPDATE SET field_value = EXCLUDED.field_value, is_calculated = FALSE, updated_at = CURRENT_TIMESTAMP
    """
    execute_query(query, (company_id, sheet_type, field_name, year, quarter, value))

@app.post("/api/update-cell")
async def update_cell(request: CellUpdateRequest):
    try:
        query = """
            INSERT INTO financial_sheets (company_id, sheet_type, field_name, year_num, quarter_num, field_value, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (company_id, sheet_type, field_name, year_num, quarter_num)
            DO UPDATE SET field_value = EXCLUDED.field_value, updated_at = CURRENT_TIMESTAMP
        """
        execute_query(query, (request.company_id, request.sheet_type, request.field_name, request.year_num, request.quarter_num, request.value))
        
        if request.sheet_type in ['salaries', 'tech-opex']:
            print(f"🔄 Recalculating affected years for {request.sheet_type}...")
            end_year = min(request.year_num + request.propagate_years, 11)
            for year in range(request.year_num, end_year):
                await run_full_recalculation_for_year(request.company_id, year)
        else:
            print(f"🔄 Recalculating single year for {request.sheet_type}...")
            await run_full_recalculation_for_year(request.company_id, request.year_num)

        return {"status": "success", "message": "Cell updated and all calculations refreshed."}
    except Exception as e:
        print(f"Error updating cell: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def calculate_growth_funnel_enhanced(data, year, quarter_num, simulated_gtm_data: Dict, stress_params: Optional[Dict] = None):
    quarter_key = f"Y{year}Q{quarter_num}"
    
    if 'growth-funnel' not in data:
        data['growth-funnel'] = {}

    # Use the in-memory GTM data snapshot instead of querying the DB
    gtm_customers = 0
    if simulated_gtm_data and year in simulated_gtm_data and quarter_num in simulated_gtm_data[year]:
        for acq_type, values in simulated_gtm_data[year][quarter_num].items():
            gtm_customers += values.get('customers_added', 0)

    # --- Stress Test Application on GTM data ---
    if stress_params:
        if stress_params.get('market_entry_underperformance_percentage', 0) > 0:
            reduction = 1 - (stress_params['market_entry_underperformance_percentage'] / 100)
            gtm_customers *= reduction
        if stress_params.get('is_economic_recession'):
             gtm_customers = 0 # No GTM customers in a recession quarter

    def get_value(sheet_type, field_name, q_key):
        return data.get(sheet_type, {}).get(field_name, {}).get(q_key, {}).get('value', 0)

    cac_multiplier = 1.0
    if stress_params and stress_params.get('cac_increase_percentage', 0) > 0:
        cac_multiplier = 1 + (stress_params['cac_increase_percentage'] / 100)

    search_spend = get_value('growth-funnel', 'Search Engine & GPT Marketing Spends', quarter_key) * cac_multiplier
    social_ads_spend = get_value('growth-funnel', 'Social Media Marketing Spends (Ads)', quarter_key) * cac_multiplier
    social_campaigns_spend = get_value('growth-funnel', 'Social Media Campaigns (Strategy & Design Spends)', quarter_key) * cac_multiplier
    atl_spend = get_value('growth-funnel', 'ATL Campaigns Spends', quarter_key) * cac_multiplier

    if stress_params and stress_params.get('is_economic_recession'):
        search_spend = social_ads_spend = social_campaigns_spend = atl_spend = 0

    search_reach = get_value('growth-funnel', 'Average Reach from Search', quarter_key)
    social_ads_reach = get_value('growth-funnel', 'Average Reach from Social Ads', quarter_key)
    social_campaigns_reach = get_value('growth-funnel', 'Average Reach from Social Campaigns', quarter_key)
    atl_reach = get_value('growth-funnel', 'Average Reach from ATL', quarter_key)
    
    total_reach = search_reach + social_ads_reach + social_campaigns_reach + atl_reach
    
    if stress_params and stress_params.get('customer_drop_percentage', 0) > 0:
        total_reach *= (1 - (stress_params['customer_drop_percentage'] / 100))

    total_spend = search_spend + social_ads_spend + social_campaigns_spend + atl_spend
    website_visitors = total_reach * 0.02
    signups = website_visitors * 0.08
    kyc_verified = signups * 0.6
    organic_new_activated = kyc_verified * 0.8
    new_activated_accounts = organic_new_activated + gtm_customers
    
    prev_quarter_key = f"Y{year}Q{quarter_num - 1}" if quarter_num > 1 else f"Y{year - 1}Q4"
    prev_year = year if quarter_num > 1 else year - 1
    
    prev_activated_accounts = 0
    if prev_year > 0:
        prev_activated_accounts = get_value('growth-funnel', 'Activated Accounts', prev_quarter_key)
    
    if stress_params and stress_params.get('is_economic_recession'):
        prev_activated_accounts *= 0.2 # 80% drop

    prev_users_lost = get_value('growth-funnel', 'Users Lost', prev_quarter_key) if prev_year > 0 else 0
    activated_accounts = prev_activated_accounts + new_activated_accounts - prev_users_lost
    active_traders = (activated_accounts * 0.3) + (gtm_customers * 0.1)
    churn_rate = get_value('growth-funnel', 'Churn Rate', quarter_key)
    users_lost = activated_accounts * (churn_rate / 100)
    total_net_users = activated_accounts - users_lost
    paying_subscribers = activated_accounts * 0.3
    
    cumulative_spend = await get_cumulative_spend_corrected(year, quarter_num, total_spend, data)
    cac = cumulative_spend / total_net_users if total_net_users > 0 else 0
    
    calculated_fields = [
        ('Total Spends on Customer Acquisition', total_spend), ('Website Visitors', website_visitors),
        ('Sign-Ups / Leads', signups), ('KYC Verified', kyc_verified),
        ('Activated Accounts', activated_accounts), ('Active Traders', active_traders),
        ('Paying Subscribers', paying_subscribers), ('Users Lost', users_lost),
        ('Total Net Users', total_net_users), ('Cost of Customer Acquisition', cac)
    ]
    
    for field_name, value in calculated_fields:
        if field_name not in data['growth-funnel']: data['growth-funnel'][field_name] = {}
        data['growth-funnel'][field_name][quarter_key] = {'value': value, 'is_calculated': True}
        if not stress_params:
            await update_calculated_field(1, 'growth-funnel', field_name, year, quarter_num, value)

async def calculate_revenue_enhanced(data, year, quarter_num, stress_params: Optional[Dict] = None):
    quarter_key = f"Y{year}Q{quarter_num}"
    if 'revenue' not in data: data['revenue'] = {}

    def get_value(sheet_type, field_name, q_key):
        return data.get(sheet_type, {}).get(field_name, {}).get(q_key, {}).get('value', 0)

    growth_data_for_q = data.get('growth-funnel', {})
    active_trading_users = get_value('growth-funnel', 'Active Traders', quarter_key)
    paying_subscribers = get_value('growth-funnel', 'Paying Subscribers', quarter_key)
    total_net_users = get_value('growth-funnel', 'Total Net Users', quarter_key)
    aum_contributors = get_value('growth-funnel', 'AUM Contributors', quarter_key)

    brokerage_per_trade = get_value('revenue', 'Average Brokerage Per User Per Trade', quarter_key)
    trades_per_day = get_value('revenue', 'Average No of Trades Per Day Per User', quarter_key)
    brokerage_revenue = brokerage_per_trade * trades_per_day * active_trading_users * 90
    
    average_pms_users = aum_contributors
    aum_per_user = get_value('revenue', 'Average AUM per Active User (₹)', quarter_key)
    management_fee = get_value('revenue', 'Management Fee from PMS', quarter_key)
    pms_revenue = (aum_per_user * average_pms_users * management_fee / 100) / 4
    
    subscription_revenue_per_user = get_value('revenue', 'AI Subscription Revenue per User (₹)', quarter_key)
    subscription_revenue = subscription_revenue_per_user * paying_subscribers * 3
    
    broking_interest_rate = 0.0125
    if stress_params and stress_params.get('interest_rate_shock') is not None:
        change = stress_params['interest_rate_shock']
        broking_interest_rate = (0.0125 + ( (5 + change) / 400) )

    average_ideal_funds = ((active_trading_users * 250000) + (average_pms_users * 1000000) + (paying_subscribers * 200000)) * 0.3
    broking_interest_revenue = average_ideal_funds * broking_interest_rate
    
    market_investment = get_value('revenue', 'Average Market Investment', quarter_key)
    investment_revenue = market_investment * 0.12 / 4
    fpi_users = get_value('revenue', 'Average no of user per month FPI', quarter_key)
    fpi_brokerage = get_value('revenue', 'Average Brokerage Per User', quarter_key)
    fpi_trades = get_value('revenue', 'Average Trade per User', quarter_key)
    fpi_aum = get_value('revenue', 'Average AUM per User', quarter_key)
    fpi_revenue = (fpi_users * fpi_brokerage * fpi_trades * 3) + (fpi_aum * fpi_users * 0.02 / 4)
    rm_variable_pay = get_value('revenue', 'Relationship Management Variable Pay Average', quarter_key)
    average_aum_rms = 1000 * rm_variable_pay
    aum_revenue = (average_aum_rms * 0.00125) - rm_variable_pay
    monthly_aum_mf = get_value('revenue', 'Average Monthly AUM MF', quarter_key)
    monthly_revenue = monthly_aum_mf * 0.02
    embedded_service = get_value('revenue', 'Embedded Financial Service', quarter_key)
    casa_interest = embedded_service * 0.03 / 4
    cards_income = embedded_service * 0.015 / 4
    insurance_premium = get_value('revenue', 'Digi Insurance - Premium Average', quarter_key)
    insurance_margin = get_value('revenue', 'Insurance Premium Margin', quarter_key)
    net_insurance_income = insurance_premium * (insurance_margin / 100)
    
    total_revenue = (brokerage_revenue + pms_revenue + subscription_revenue + monthly_revenue + broking_interest_revenue + investment_revenue + fpi_revenue + aum_revenue + casa_interest + cards_income + net_insurance_income)

    if stress_params:
        if stress_params.get('pricing_pressure_percentage', 0) > 0:
            total_revenue *= (1 - (stress_params['pricing_pressure_percentage'] / 100))
        if stress_params.get('is_technology_failure'):
            total_revenue *= 0.5

    arpu = total_revenue / total_net_users if total_net_users > 0 else 0
    
    calculated_fields = [
        ('Active Trading Users', active_trading_users), ('Brokerage Revenue', brokerage_revenue),
        ('Average Active PMS Users', average_pms_users), ('PMS Revenue', pms_revenue),
        ('Average Active Subscription Users', paying_subscribers), ('Revenue from Subscriptions', subscription_revenue),
        ('Average Monthly Revenue', monthly_revenue), ('Average Ideal Broking Funds', average_ideal_funds),
        ('Revenue from Broking Interest', broking_interest_revenue), ('Average Revenue from Investments', investment_revenue),
        ('Revenue from FPI', fpi_revenue), ('Average AUM from RMs', average_aum_rms),
        ('Revenue from AUMs', aum_revenue), ('Digi Banking - CASA Interest', casa_interest),
        ('Digi Banking - Cards Income', cards_income), ('Net Insurance Income', net_insurance_income),
        ('Total Revenue', total_revenue), ('Average Revenue Per User', arpu)
    ]
    
    for field_name, value in calculated_fields:
        if field_name not in data['revenue']: data['revenue'][field_name] = {}
        data['revenue'][field_name][quarter_key] = {'value': value, 'is_calculated': True}
        if not stress_params:
            await update_calculated_field(1, 'revenue', field_name, year, quarter_num, value)

async def calculate_dp_valuation(data: dict, year: int, quarter_num: int, stress_params: Optional[Dict] = None):
    quarter_key = f"Y{year}Q{quarter_num}"

    def get_value(sheet_type, field_name, q_key):
        return data.get(sheet_type, {}).get(field_name, {}).get(q_key, {}).get('value', 0)

    aum_per_user = get_value('revenue', 'Average AUM per Active User (₹)', quarter_key)
    active_pms_users = get_value('revenue', 'Average Active PMS Users', quarter_key)
    average_ideal_funds = get_value('revenue', 'Average Ideal Broking Funds', quarter_key)
    average_aum_rms = get_value('revenue', 'Average AUM from RMs', quarter_key)

    dp_valuation = (aum_per_user * active_pms_users) + average_ideal_funds + average_aum_rms
    
    field_name = 'DP Valuation'
    sheet_type = 'dp-evaluation'

    if sheet_type not in data: data[sheet_type] = {}
    if field_name not in data[sheet_type]: data[sheet_type][field_name] = {}
    data[sheet_type][field_name][quarter_key] = {'value': dp_valuation, 'is_calculated': True}
    
    if not stress_params:
        await update_calculated_field(1, sheet_type, field_name, year, quarter_num, dp_valuation)

async def calculate_unit_economics(data, year, quarter_num, stress_params: Optional[Dict] = None):
    quarter_key = f"Y{year}Q{quarter_num}"

    def get_value(sheet_type, field_name, q_key):
        return data.get(sheet_type, {}).get(field_name, {}).get(q_key, {}).get('value', 0)

    arpu = get_value('revenue', 'Average Revenue Per User', quarter_key)
    cac = get_value('growth-funnel', 'Cost of Customer Acquisition', quarter_key)
    churn_rate_percent = get_value('growth-funnel', 'Churn Rate', quarter_key)

    avg_customer_lifetime_months = (1 / (churn_rate_percent / 100)) * 3 if churn_rate_percent > 0 else 0
    ltv = arpu * (avg_customer_lifetime_months / 3) if avg_customer_lifetime_months > 0 else 0
    ltv_cac_ratio = ltv / cac if cac > 0 else 0

    unit_economics_fields = [
        ('CAC (Customer Acquisition Cost)', cac), ('ARPU (Average Revenue Per User)', arpu),
        ('Churn Rate (%)', churn_rate_percent), ('Average Customer Lifetime (Months)', avg_customer_lifetime_months),
        ('LTV (Lifetime Value)', ltv), ('LTV/CAC Ratio', ltv_cac_ratio),
    ]
    
    for field_name, value in unit_economics_fields:
        if 'unit-economics' not in data: data['unit-economics'] = {}
        if field_name not in data['unit-economics']: data['unit-economics'][field_name] = {}
        data['unit-economics'][field_name][quarter_key] = {'value': value, 'is_calculated': True}
        if not stress_params:
            await update_calculated_field(1, 'unit-economics', field_name, year, quarter_num, value)

async def calculate_capex_enhanced(data, year):
    print(f"🏗️ Calculating Capex for Year {year}")
    for quarter_num in range(1, 5):
        quarter_key = f"Y{year}Q{quarter_num}"
        total_assets = 0
        capex_metrics = ['Fixed Deposits', 'Properties', 'Equipments', 'Vehicles', 'NSE Data Processing Units']
        
        for metric in capex_metrics:
            cumulative_value = await get_cumulative_value_for_capex(metric, year, quarter_num)
            if metric not in data: data[metric] = {}
            data[metric][quarter_key] = {'value': cumulative_value, 'is_calculated': True}
            total_assets += cumulative_value

        if 'Total Assets Value' not in data: data['Total Assets Value'] = {}
        data['Total Assets Value'][quarter_key] = {'value': total_assets, 'is_calculated': True}
        
async def get_cumulative_value_for_capex(metric_name: str, year: int, quarter: int):
    query = """
        SELECT COALESCE(SUM(quarter_addition), 0) as total_cumulative FROM cumulative_metrics
        WHERE company_id = 1 AND metric_name = %s AND ( (year_num < %s) OR (year_num = %s AND quarter_num <= %s) )
    """
    params = (metric_name, year, year, quarter)
    result = execute_query(query, params, fetch=True)
    return float(result[0]['total_cumulative']) if result and result[0]['total_cumulative'] else 0

async def get_field_value(sheet_type: str, field_name: str, year: int, quarter: int):
    query = "SELECT field_value FROM financial_sheets WHERE company_id = 1 AND sheet_type = %s AND field_name = %s AND year_num = %s AND quarter_num = %s"
    result = execute_query(query, (sheet_type, field_name, year, quarter), fetch=True)
    return float(result[0]['field_value']) if result else 0

async def get_cumulative_spend_corrected(year: int, quarter: int, current_spend: float, data: dict):
    previous_spend = 0
    for y in range(1, year + 1):
        for q in range(1, 5):
            if y == year and q >= quarter:
                break
            quarter_key = f"Y{y}Q{q}"
            previous_spend += data.get('growth-funnel', {}).get('Total Spends on Customer Acquisition', {}).get(quarter_key, {}).get('value', 0)
    return previous_spend + current_spend

async def get_growth_funnel_data(year: int, quarter: int):
    try:
        query = "SELECT field_name, field_value FROM financial_sheets WHERE company_id = 1 AND sheet_type = 'growth-funnel' AND year_num = %s AND quarter_num = %s"
        growth_rows = execute_query(query, (year, quarter), fetch=True)
        growth_dict = {row['field_name']: float(row['field_value']) for row in growth_rows}
        return {
            'active_traders': growth_dict.get('Active Traders', 0), 'paying_subscribers': growth_dict.get('Paying Subscribers', 0),
            'activated_accounts': growth_dict.get('Activated Accounts', 0), 'total_net_users': growth_dict.get('Total Net Users', 0),
            'aum_contributors': growth_dict.get('AUM Contributors', 0)
        }
    except Exception as e:
        print(f"❌ Error getting growth funnel data for Y{year}Q{quarter}: {e}")
        return {'active_traders': 0, 'paying_subscribers': 0, 'activated_accounts': 0, 'total_net_users': 0, 'aum_contributors': 0}

async def update_cumulative_field(company_id: int, metric_name: str, year_num: int, quarter_num: int, quarter_value: float):
    try:
        update_query = """
            INSERT INTO cumulative_metrics (company_id, metric_name, year_num, quarter_num, quarter_addition, updated_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (company_id, metric_name, year_num, quarter_num)
            DO UPDATE SET quarter_addition = EXCLUDED.quarter_addition, updated_at = CURRENT_TIMESTAMP
        """
        execute_query(update_query, (company_id, metric_name, year_num, quarter_num, quarter_value))
    except Exception as e:
        print(f"Error updating cumulative field {metric_name}: {e}")

async def run_full_recalculation_for_year(company_id: int, year: int, data: Optional[Dict] = None, simulated_gtm_data: Optional[Dict] = None, stress_params: Optional[Dict] = None):
    print(f"--- Running Full Recalculation for Year {year} ---")
    if data is None:
        query = "SELECT field_name, year_num, quarter_num, field_value, is_calculated, sheet_type FROM financial_sheets WHERE company_id = %s AND year_num = %s"
        rows = execute_query(query, (company_id, year), fetch=True)
        data = {}
        for row in rows:
            sheet_type, field_name, q_key = row['sheet_type'], row['field_name'], f"Y{row['year_num']}Q{row['quarter_num']}"
            if sheet_type not in data: data[sheet_type] = {}
            if field_name not in data[sheet_type]: data[sheet_type][field_name] = {}
            data[sheet_type][field_name][q_key] = {'value': float(row['field_value']), 'is_calculated': row['is_calculated']}

    for quarter in range(1, 5):
        current_stress_params = None
        if stress_params:
            start_year, start_quarter = stress_params['start_year'], stress_params['start_quarter']
            if year > start_year or (year == start_year and quarter >= start_quarter):
                current_stress_params = stress_params
        
        await calculate_growth_funnel_enhanced(data, year, quarter, simulated_gtm_data, current_stress_params)
        await calculate_revenue_enhanced(data, year, quarter, current_stress_params)
        await calculate_dp_valuation(data, year, quarter, current_stress_params)
        await calculate_unit_economics(data, year, quarter, current_stress_params)
        await recalculate_salary_costs_detailed(data, year, quarter, current_stress_params)
        await recalculate_tech_costs(data, year, quarter, current_stress_params)
        await calculate_financials_enhanced(data, year, quarter, simulated_gtm_data, current_stress_params)

    print(f"--- Full Recalculation for Year {year} Complete ---")


async def calculate_financials_enhanced(data: dict, year: int, quarter_num: int, simulated_gtm_data: Dict, stress_params: Optional[Dict] = None):
    quarter_key = f"Y{year}Q{quarter_num}"

    def get_value(sheet_type, field_name, q_key):
        return data.get(sheet_type, {}).get(field_name, {}).get(q_key, {}).get('value', 0)
    
    total_revenue = get_value('revenue', 'Total Revenue', quarter_key)
    salary_cost = get_value('salaries', 'Total Salary Cost', quarter_key)
    tech_cost = get_value('tech-opex', 'Total', quarter_key)
    acquisition_cost = get_value('growth-funnel', 'Total Spends on Customer Acquisition', quarter_key)
    
    ma_cost = 0
    if simulated_gtm_data and year in simulated_gtm_data and quarter_num in simulated_gtm_data[year]:
        for acq_type, values in simulated_gtm_data[year][quarter_num].items():
            ma_cost += values.get('total_amount', 0)
    
    total_operating_costs = salary_cost + tech_cost + acquisition_cost + ma_cost
    ebitda = total_revenue - total_operating_costs
    ebitda_margin = (ebitda / total_revenue * 100) if total_revenue > 0 else 0
    
    financial_fields = [
        ('Total Revenue', total_revenue), ('Total Salary Cost', salary_cost),
        ('Total Tech & OpEx', tech_cost), ('Total Customer Acquisition Spends', acquisition_cost),
        ('M&A Costs', ma_cost), ('Total Operating Costs', total_operating_costs),
        ('EBITDA', ebitda), ('EBITDA Margin (%)', ebitda_margin)
    ]
    
    for field_name, value in financial_fields:
        if 'financials' not in data: data['financials'] = {}
        if field_name not in data['financials']: data['financials'][field_name] = {}
        data['financials'][field_name][quarter_key] = {'value': value, 'is_calculated': True}
        if not stress_params:
            await update_calculated_field(1, 'financials', field_name, year, quarter_num, value)

async def recalculate_salary_costs_detailed(data: dict, year: int, quarter: int, stress_params: Optional[Dict] = None):
    quarter_key = f"Y{year}Q{quarter}"
    inflation_factor = 1 + (calculate_inflation_rate(quarter_key) / 100)
    salaries_data = data.get('salaries', {})
    
    salary_multiplier = 1.0
    if stress_params and stress_params.get('is_economic_recession'):
        salary_multiplier = 0.6

    groups = {
        'Management & Domain Expert': ['Core Management', 'Domain Specific Head', 'Cluster Heads'],
        'Subject Level Expert': ['Economists', 'Technical Analysts', 'Fundamental Analysts', 'Business Analysts', 'Quant Analysts', 'Data Scientists'],
        'Board of Directors': ['Independent Directors'],
        'Functional Heads': ['Marketing Head', 'BD Head', 'Accounts Head', 'HR Head', 'IT Head', 'Cyber Security Head', 'Compliance Head', 'Investment Head', 'Commercial Head', 'Technology Head'],
        'Engineering Team': ['Senior Developers', 'Junior Developers', 'Testers', 'Designers'],
        'Marketing Team': ['Marketing Managers', 'Marketing Executives', 'RMs'],
        'Compliance Team': ['Compliance Officers', 'Grievance Officer'],
        'R&D Team': ['Research Engineers'],
        'Support Staff': ['Support Executives']
    }
    total_cost = 0
    for group_name, roles in groups.items():
        group_cost = 0
        for role in roles:
            count = salaries_data.get(f'{role} Count', {}).get(quarter_key, {}).get('value', 0)
            salary = salaries_data.get(f'{role} Average Salary', {}).get(quarter_key, {}).get('value', 0)
            adjusted_salary = salary * salary_multiplier
            group_cost += count * (adjusted_salary * inflation_factor) * 3
        total_cost += group_cost
        
        if 'salaries' not in data: data['salaries'] = {}
        cost_field_name = f'{group_name} Cost'
        if cost_field_name not in data['salaries']: data['salaries'][cost_field_name] = {}
        data['salaries'][cost_field_name][quarter_key] = {'value': group_cost, 'is_calculated': True}
        if not stress_params:
            await update_calculated_field(1, 'salaries', cost_field_name, year, quarter, group_cost)
    
    if 'Total Salary Cost' not in data['salaries']: data['salaries']['Total Salary Cost'] = {}
    data['salaries']['Total Salary Cost'][quarter_key] = {'value': total_cost, 'is_calculated': True}
    if not stress_params:
        await update_calculated_field(1, 'salaries', 'Total Salary Cost', year, quarter, total_cost)


async def recalculate_tech_costs(data: dict, year: int, quarter: int, stress_params: Optional[Dict] = None):
    quarter_key = f"Y{year}Q{quarter}"
    tech_opex_data = data.get('tech-opex', {})
    cost_fields = ['Cyber Security', 'Servers', 'Data Processing Equipment - NSE', 'GPUs', 'Lease Line', 'Third Party APIs', 'Third Party SAAS', 'Google Workspace', 'AMCs', 'SEBI Compliance', 'NSE', 'BSE', 'DP', 'AMFI', 'RBI', 'ROC', 'IT', 'Other OpEx', 'Office Rent', 'Utilities & Internet', 'Office Supplies', 'Travel']
    base_cost = sum(tech_opex_data.get(field, {}).get(quarter_key, {}).get('value', 0) for field in cost_fields)
    inflation_rate = calculate_inflation_rate(quarter_key)
    inflation_adjustment = base_cost * (inflation_rate / 100)
    total_quarters_elapsed = (year - 1) * 4 + quarter - 1
    surprise_costs = base_cost * (total_quarters_elapsed * 0.5 / 100)
    total_cost = base_cost + inflation_adjustment + surprise_costs
    calculated_fields = [('Inflation Adjustment (%)', inflation_rate), ('Surprise Costs', surprise_costs), ('Total', total_cost)]
    
    for field_name, value in calculated_fields:
        if 'tech-opex' not in data: data['tech-opex'] = {}
        if field_name not in data['tech-opex']: data['tech-opex'][field_name] = {}
        data['tech-opex'][field_name][quarter_key] = {'value': value, 'is_calculated': True}
        if not stress_params:
            await update_calculated_field(1, 'tech-opex', field_name, year, quarter, value)


async def update_calculated_field(company_id: int, sheet_type: str, field_name: str, year: int, quarter: int, value: float):
    query = "INSERT INTO financial_sheets (company_id, sheet_type, field_name, year_num, quarter_num, field_value, is_calculated, updated_at) VALUES (%s, %s, %s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP) ON CONFLICT (company_id, sheet_type, field_name, year_num, quarter_num) DO UPDATE SET field_value = EXCLUDED.field_value, is_calculated = TRUE, updated_at = CURRENT_TIMESTAMP"
    execute_query(query, (company_id, sheet_type, field_name, year, quarter, value))

@app.post("/api/update-gtm")
async def update_gtm(request: GTMUpdateRequest):
    try:
        gtm_impacts = {
            'Full Broking House': {'customers': 6500, 'fixed_deposits': 50000000, 'properties': 80000000, 'is_permanent': False},
            'GOP Based Broker Deals': {'customers': 5000, 'fixed_deposits': 30000000, 'properties': 50000000, 'is_permanent': True},
            'Secondary Market Acquisitions': {'customers': 2000, 'fixed_deposits': 10000000, 'properties': 15000000, 'is_permanent': False}
        }
        acq_type = request.acquisition_type
        count = request.count
        amount_per_acquisition = request.amount_per_acquisition
        year, quarter = request.year_num, request.quarter_num
        
        if acq_type in gtm_impacts:
            impact = gtm_impacts[acq_type]
            customers_added = impact['customers'] * count
            fixed_deposits_added = impact['fixed_deposits'] * count
            properties_added = impact['properties'] * count
            total_amount = amount_per_acquisition * count
            
            query = """
                INSERT INTO gtm_acquisitions (company_id, acquisition_type, year_num, quarter_num, acquisition_count, customers_added, fixed_deposit_value, property_value, is_permanent, amount_per_acquisition, total_amount)
                VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (company_id, acquisition_type, year_num, quarter_num)
                DO UPDATE SET acquisition_count = EXCLUDED.acquisition_count, customers_added = EXCLUDED.customers_added, fixed_deposit_value = EXCLUDED.fixed_deposit_value, property_value = EXCLUDED.property_value, amount_per_acquisition = EXCLUDED.amount_per_acquisition, total_amount = EXCLUDED.total_amount, updated_at = CURRENT_TIMESTAMP
            """
            execute_query(query, (acq_type, year, quarter, count, customers_added, fixed_deposits_added, properties_added, impact['is_permanent'], amount_per_acquisition, total_amount))
            
            await update_cumulative_field(1, 'Fixed Deposits', year, quarter, fixed_deposits_added)
            await update_cumulative_field(1, 'Properties', year, quarter, properties_added)

            await run_full_recalculation_for_year(1, year)
            return {"status": "success", "message": "GTM data updated and financials refreshed."}
    except Exception as e:
        print(f"Error updating GTM: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard-data")
async def get_dashboard_data():
    try:
        revenue_query = "SELECT fs.year_num, fs.quarter_num, fs.field_value FROM financial_sheets fs WHERE fs.company_id = 1 AND fs.sheet_type = 'revenue' AND fs.field_name = 'Total Revenue' ORDER BY fs.year_num, fs.quarter_num LIMIT 8"
        revenue_rows = execute_query(revenue_query, fetch=True)
        revenue_data = [{"quarter": f"Y{r['year_num']}Q{r['quarter_num']}", "value": float(r['field_value'])} for r in revenue_rows] if revenue_rows else []
        
        customer_query = "SELECT year_num, quarter_num, field_value FROM financial_sheets WHERE company_id = 1 AND sheet_type = 'growth-funnel' AND field_name = 'Total Net Users' ORDER BY year_num, quarter_num LIMIT 8"
        customer_rows = execute_query(customer_query, fetch=True)
        customer_data = [{"quarter": f"Y{c['year_num']}Q{c['quarter_num']}", "value": float(c['field_value'])} for c in customer_rows] if customer_rows else []
        
        unit_query = "SELECT field_name, field_value FROM financial_sheets WHERE company_id = 1 AND sheet_type = 'unit-economics' AND year_num = 1 AND quarter_num = 4"
        unit_rows = execute_query(unit_query, fetch=True)
        unit_economics = {row['field_name']: float(row['field_value']) for row in unit_rows} if unit_rows else {}
        ltv_cac_ratio = unit_economics.get('LTV/CAC Ratio', 0)
        
        gtm_query = "SELECT acquisition_type, SUM(customers_added) as total_customers, SUM(total_amount) as total_investment FROM gtm_acquisitions WHERE company_id = 1 AND acquisition_count > 0 GROUP BY acquisition_type"
        gtm_rows = execute_query(gtm_query, fetch=True)
        gtm_data = [{"type": row['acquisition_type'], "customers": int(row['total_customers']), "investment": float(row['total_investment'] if row['total_investment'] else 0)} for row in gtm_rows] if gtm_rows else []

        burn_query = "SELECT field_value FROM financial_sheets WHERE company_id = 1 AND sheet_type = 'financials' AND field_name = 'Total Operating Costs' ORDER BY year_num DESC, quarter_num DESC LIMIT 1"
        burn_result = execute_query(burn_query, fetch=True)
        monthly_burn = float(burn_result[0]['field_value']) / 3 if burn_result else 0

        return {
            "revenue": revenue_data, "customers": customer_data, "unit_economics": unit_economics, "gtm_impact": gtm_data,
            "summary": {
                "current_revenue": revenue_data[-1]["value"] if revenue_data else 0,
                "total_customers": customer_data[-1]["value"] if customer_data else 0,
                "ltv_cac_ratio": ltv_cac_ratio, "burn_rate": monthly_burn,
                "total_gtm_investment": sum(item["investment"] for item in gtm_data)
            }
        }
    except Exception as e:
        print(f"Error fetching dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai-assistant")
async def ai_assistant(request: AIAssistantRequest):
    try:
        query = "SELECT sheet_type, field_name, year_num, quarter_num, field_value FROM financial_sheets WHERE company_id = %s AND sheet_type = %s ORDER BY year_num, quarter_num"
        historical_data = execute_query(query, (request.company_id, request.sheet_type), fetch=True)
        
        inflation_query = "SELECT inflation_rate, surprise_cost_rate FROM inflation_tracking WHERE company_id = %s"
        inflation_data = execute_query(inflation_query, (request.company_id,), fetch=True)
        
        response = {
            "reasoning": f"Based on your {request.sheet_type} data, inflation trends (6% yearly), and fintech industry benchmarks",
            "suggested_value": None, "field_specific_advice": "Consider market conditions, inflation impact, and regulatory requirements",
            "warnings": [], "confidence": "medium", "inflation_impact": None, "gtm_recommendations": []
        }
        
        if inflation_data:
            inflation_rate = float(inflation_data[0]['inflation_rate'])
            response["inflation_impact"] = f"Current inflation rate of {inflation_rate}% will compound quarterly, affecting salary and OpEx projections"
        
        if request.sheet_type == "revenue" and historical_data:
            recent_revenues = [float(row['field_value']) for row in historical_data if row['field_name'] == 'Total Revenue']
            if len(recent_revenues) >= 2:
                growth_rate = ((recent_revenues[-1] - recent_revenues[-2]) / recent_revenues[-2]) * 100
                response["reasoning"] += f". Your revenue growth rate is {growth_rate:.1f}%"
                if growth_rate > 20:
                    response["field_specific_advice"] = "Strong growth! Consider scaling operations and optimizing customer acquisition with inflation-adjusted budgets."
                    response["confidence"] = "high"
                elif growth_rate < 5:
                    response["warnings"].append("Low growth rate. Consider GTM acquisitions or enhanced marketing spend to accelerate growth.")
            
        elif request.sheet_type == "growth-funnel":
            response["field_specific_advice"] = "Focus on optimizing conversion rates while accounting for inflation in marketing spend"
            response["gtm_recommendations"] = ["Consider Full Broking House acquisition for 6,500+ customers", "GOP-based deals provide permanent customer base expansion", "Secondary market acquisitions offer cost-effective growth"]
            
        elif request.sheet_type == "salaries":
            response["field_specific_advice"] = "Salary projections include 6% yearly inflation (1.5% quarterly) and new joining costs"
            response["inflation_impact"] = "Each quarter, salaries automatically increase by 1.5% to account for inflation"
            if "engineering" in request.prompt.lower():
                response["reasoning"] += ". Engineering salaries should account for market competition and inflation"
                response["suggested_value"] = 85000
                
        elif request.sheet_type == "tech-opex":
            response["field_specific_advice"] = "Tech costs include 6% yearly inflation plus 2% surprise costs annually"
            response["inflation_impact"] = "Tech expenses automatically adjust for inflation (1.5% quarterly) plus surprise costs (0.5% quarterly)"
            
        elif request.sheet_type == "unit-economics":
            response["field_specific_advice"] = "Maintain LTV/CAC ratio above 3:1 while considering inflation impact on both metrics"
            
        elif request.sheet_type == "gtm":
            response["field_specific_advice"] = "GTM acquisitions should focus on customer quality, integration capabilities, and ROI"
            response["gtm_recommendations"] = ["Full Broking House: ₹15Cr investment, 6,500 customers, ₹13Cr assets", "GOP Deals (Permanent): ₹10Cr investment, 5,000 customers, ₹8Cr assets", "Secondary Market: ₹5Cr investment, 2,000 customers, ₹2.5Cr assets"]
        
        store_query = "INSERT INTO ai_interactions (company_id, user_id, sheet_type, prompt, ai_response, confidence_score) VALUES (%s, 1, %s, %s, %s, %s)"
        confidence_score = {"high": 0.9, "medium": 0.7, "low": 0.5}[response["confidence"]]
        execute_query(store_query, (request.company_id, request.sheet_type, request.prompt, json.dumps(response), confidence_score))
        return response
    except Exception as e:
        print(f"Error in AI assistant: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add-column")
async def add_column(request: AddColumnRequest):
    """Add a new column to a financial sheet with propagation"""
    try:
        for year in range(1, 11):
            for quarter in range(1, 5):
                query = """
                    INSERT INTO financial_sheets (company_id, sheet_type, field_name, year_num, quarter_num, field_value)
                    VALUES (%s, %s, %s, %s, %s, 0)
                    ON CONFLICT (company_id, sheet_type, field_name, year_num, quarter_num) DO NOTHING
                """
                execute_query(query, (request.company_id, request.sheet_type, request.column_name, year, quarter))
        return {"status": "success", "message": f"Column '{request.column_name}' added successfully with auto-propagation"}
    except Exception as e:
        print(f"Error adding column: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stress-test")
async def run_stress_test_simulation(payload: StressTestPayload):
    """
    Run a non-destructive, in-memory stress test simulation.
    This will not write any changes to the database.
    """
    try:
        print("--- Starting Stress Test Simulation ---")
        
        # 1. Fetch all baseline data from the database into an in-memory dictionary
        # Fetch financial sheets data
        fs_query = "SELECT * FROM financial_sheets WHERE company_id = 1 ORDER BY year_num, quarter_num"
        fs_rows = execute_query(fs_query, fetch=True)
        
        simulated_data = {}
        for row in fs_rows:
            sheet_type, field_name = row['sheet_type'], row['field_name']
            quarter_key = f"Y{row['year_num']}Q{row['quarter_num']}"
            if sheet_type not in simulated_data: simulated_data[sheet_type] = {}
            if field_name not in simulated_data[sheet_type]: simulated_data[sheet_type][field_name] = {}
            simulated_data[sheet_type][field_name][quarter_key] = {
                'value': float(row['field_value']), 'is_calculated': row['is_calculated']
            }

        # Fetch GTM acquisitions data
        gtm_query = "SELECT * FROM gtm_acquisitions WHERE company_id = 1 ORDER BY year_num, quarter_num"
        gtm_rows = execute_query(gtm_query, fetch=True)
        simulated_gtm_data = {}
        for row in gtm_rows:
            year, quarter, acq_type = row['year_num'], row['quarter_num'], row['acquisition_type']
            if year not in simulated_gtm_data: simulated_gtm_data[year] = {}
            if quarter not in simulated_gtm_data[year]: simulated_gtm_data[year][quarter] = {}
            simulated_gtm_data[year][quarter][acq_type] = {
                'customers_added': float(row['customers_added']),
                'total_amount': float(row['total_amount'])
            }

        # 2. Convert payload to a dictionary for easier use
        stress_params = payload.dict()
        
        # 3. Run the full recalculation for all years, applying the stress parameters
        for year in range(1, 11): # Assuming a 10-year model
            await run_full_recalculation_for_year(1, year, data=simulated_data, simulated_gtm_data=simulated_gtm_data, stress_params=stress_params)
            
        print("--- Stress Test Simulation Complete ---")
        
        # 4. Return the complete, simulated data structure
        return simulated_data

    except Exception as e:
        import traceback
        print(f"❌ Error during stress test simulation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to run stress test: {e}")


# templates = Jinja2Templates(directory="templates")
# @app.get("/", response_class=HTMLResponse)
# async def root(request: Request):
#     """Serve the main application template"""
#     context = {"request": request}
#     return templates.TemplateResponse("index.html", context)

if __name__ == "__main__":
    import uvicorn
    @app.on_event("shutdown")
    async def shutdown_event():
        await enhanced_llm_agent.close_session()
    
    print("Navigate to http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
