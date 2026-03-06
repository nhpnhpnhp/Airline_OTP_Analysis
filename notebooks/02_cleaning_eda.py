# %% [markdown]
# # 🧹 02 — Data Cleaning & EDA
# **Mục tiêu**: Clean dữ liệu, EDA toàn diện, visualizations (15-20+ charts).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.figsize': (12, 6), 'font.size': 12})

# %%
# Load dữ liệu
df = pd.read_parquet("../data/processed/flights_jan_2021_2025.parquet")
print(f"Shape: {df.shape}")
print(f"Years: {sorted(df['YEAR'].unique())}")

# %% [markdown]
# ## 2.1 Data Cleaning

# %%
# --- Xử lý Cancelled & Diverted ---
print(f"Cancelled: {df['CANCELLED'].sum():,} ({df['CANCELLED'].mean()*100:.2f}%)")
print(f"Diverted:  {df['DIVERTED'].sum():,} ({df['DIVERTED'].mean()*100:.2f}%)")

# Với cancelled: delay cols → NaN
mask_cancel = df['CANCELLED'] == 1
delay_cols = ['DEP_TIME','DEP_DELAY','DEP_DEL15','ARR_TIME','ARR_DELAY','ARR_DEL15',
              'TAXI_OUT','TAXI_IN','ACTUAL_ELAPSED_TIME','AIR_TIME',
              'CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
for col in delay_cols:
    if col in df.columns:
        df.loc[mask_cancel, col] = np.nan

# %%
# --- Convert HHMM → Minutes ---
def hhmm_to_minutes(s):
    s = pd.to_numeric(s, errors='coerce')
    return (s // 100) * 60 + (s % 100)

for col in ['CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME', 'ARR_TIME']:
    if col in df.columns:
        df[col + '_MIN'] = hhmm_to_minutes(df[col])

# %%
# --- Tạo time features ---
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
df['DAY_OF_MONTH'] = df['FL_DATE'].dt.day
df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
df['ROUTE'] = df['ORIGIN'] + '-' + df['DEST']

# %%
# --- Delay category ---
bins = [-np.inf, 0, 15, 60, 120, np.inf]
labels = ['On-time/Early', 'Slight (1-15)', 'Late (16-60)', 'Very Late (61-120)', 'Extreme (>120)']
df['DELAY_CAT'] = pd.cut(df['ARR_DELAY'], bins=bins, labels=labels)

# %%
# --- Tách datasets ---
df_operated = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
print(f"Full dataset:       {len(df):,}")
print(f"Operated flights:   {len(df_operated):,}")

# %% [markdown]
# ## 2.2 OTP Analysis

# %%
# --- Chart 1: OTP Overall theo năm (Bar chart) ---
otp_year = df_operated.groupby('YEAR')['ARR_DEL15'].agg(['mean', 'count']).reset_index()
otp_year['OTP'] = 1 - otp_year['mean']  # OTP = 1 - delay rate

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(otp_year['YEAR'].astype(str), otp_year['OTP'] * 100, 
              color=['#2ecc71','#3498db','#9b59b6','#e74c3c','#f39c12'], edgecolor='white', linewidth=2)
for bar, val in zip(bars, otp_year['OTP']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('On-Time Performance (%)')
ax.set_title('OTP tháng 1 theo năm (2021–2025)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('../reports/figures/01_otp_by_year.png', dpi=150)
plt.show()

# %%
# --- Chart 2: OTP theo Carrier (Top/Bottom 10) ---
carrier_otp = (df_operated.groupby('OP_CARRIER')
               .agg(flights=('ARR_DEL15','count'), delay_rate=('ARR_DEL15','mean'))
               .reset_index())
carrier_otp['OTP'] = 1 - carrier_otp['delay_rate']
carrier_otp = carrier_otp[carrier_otp['flights'] >= 1000]  # min flights filter

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# Top 10
top10 = carrier_otp.nlargest(10, 'OTP')
axes[0].barh(top10['OP_CARRIER'], top10['OTP']*100, color='#2ecc71')
axes[0].set_title('Top 10 Carriers by OTP')
axes[0].set_xlabel('OTP (%)')
# Bottom 10
bot10 = carrier_otp.nsmallest(10, 'OTP')
axes[1].barh(bot10['OP_CARRIER'], bot10['OTP']*100, color='#e74c3c')
axes[1].set_title('Bottom 10 Carriers by OTP')
axes[1].set_xlabel('OTP (%)')
plt.tight_layout()
plt.savefig('../reports/figures/02_otp_by_carrier.png', dpi=150)
plt.show()

# %%
# --- Chart 3: OTP theo Airport Origin (Top 20) ---
origin_otp = (df_operated.groupby('ORIGIN')
              .agg(flights=('ARR_DEL15','count'), delay_rate=('ARR_DEL15','mean'))
              .reset_index())
origin_otp['OTP'] = 1 - origin_otp['delay_rate']
top20_origins = origin_otp.nlargest(20, 'flights')

fig, ax = plt.subplots(figsize=(14, 8))
colors = plt.cm.RdYlGn(top20_origins['OTP'])
ax.barh(top20_origins['ORIGIN'], top20_origins['OTP']*100, color=colors)
ax.set_xlabel('OTP (%)')
ax.set_title('OTP theo Origin Airport (Top 20 busiest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/03_otp_by_origin.png', dpi=150)
plt.show()

# %%
# --- Chart 4: OTP theo Dest Airport (Top 20) ---
dest_otp = (df_operated.groupby('DEST')
            .agg(flights=('ARR_DEL15','count'), delay_rate=('ARR_DEL15','mean'))
            .reset_index())
dest_otp['OTP'] = 1 - dest_otp['delay_rate']
top20_dest = dest_otp.nlargest(20, 'flights')

fig, ax = plt.subplots(figsize=(14, 8))
colors = plt.cm.RdYlGn(top20_dest['OTP'])
ax.barh(top20_dest['DEST'], top20_dest['OTP']*100, color=colors)
ax.set_xlabel('OTP (%)')
ax.set_title('OTP theo Destination Airport (Top 20 busiest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/04_otp_by_dest.png', dpi=150)
plt.show()

# %%
# --- Chart 5: OTP theo Route (Top 20 tuyến nhiều chuyến) ---
route_otp = (df_operated.groupby('ROUTE')
             .agg(flights=('ARR_DEL15','count'), delay_rate=('ARR_DEL15','mean'))
             .reset_index())
route_otp['OTP'] = 1 - route_otp['delay_rate']

fig, axes = plt.subplots(1, 2, figsize=(18, 10))
# Busiest routes
top_routes = route_otp.nlargest(20, 'flights')
axes[0].barh(top_routes['ROUTE'], top_routes['OTP']*100, color='#3498db')
axes[0].set_title('OTP — Top 20 Busiest Routes')
axes[0].set_xlabel('OTP (%)')
# Most delayed routes (min 500 flights)
worst_routes = route_otp[route_otp['flights'] >= 500].nsmallest(20, 'OTP')
axes[1].barh(worst_routes['ROUTE'], worst_routes['OTP']*100, color='#e74c3c')
axes[1].set_title('OTP — 20 Most Delayed Routes (min 500 flights)')
axes[1].set_xlabel('OTP (%)')
plt.tight_layout()
plt.savefig('../reports/figures/05_otp_by_route.png', dpi=150)
plt.show()

# %%
# --- Chart 6: OTP Heatmap (Year x DEP_TIME_BLK) ---
heatmap_data = (df_operated.groupby(['YEAR', 'DEP_TIME_BLK'])['ARR_DEL15']
                .mean().unstack(fill_value=0))
heatmap_data = 1 - heatmap_data  # Convert to OTP

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(heatmap_data * 100, annot=True, fmt='.0f', cmap='RdYlGn',
            ax=ax, linewidths=0.5, vmin=60, vmax=95)
ax.set_title('OTP (%) Heatmap: Year × Departure Time Block', fontsize=14, fontweight='bold')
ax.set_ylabel('Year')
ax.set_xlabel('Departure Time Block')
plt.tight_layout()
plt.savefig('../reports/figures/06_otp_heatmap.png', dpi=150)
plt.show()

# %% [markdown]
# ## 2.3 Delay Distribution

# %%
# --- Chart 7: ARR_DELAY Distribution (Histogram) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Full distribution
df_operated['ARR_DELAY'].clip(-60, 200).hist(bins=100, ax=axes[0], color='#3498db', alpha=0.7)
axes[0].axvline(x=15, color='red', linestyle='--', label='15-min threshold')
axes[0].set_title('ARR_DELAY Distribution (clipped)')
axes[0].set_xlabel('Minutes')
axes[0].legend()

# Log scale tail
df_delayed = df_operated[df_operated['ARR_DELAY'] > 15]
df_delayed['ARR_DELAY'].clip(15, 500).hist(bins=80, ax=axes[1], color='#e74c3c', alpha=0.7)
axes[1].set_title('Tail: Delay > 15 min')
axes[1].set_xlabel('Minutes')
plt.tight_layout()
plt.savefig('../reports/figures/07_delay_distribution.png', dpi=150)
plt.show()

# %%
# --- Chart 8: Box/Violin Plot — ARR_DELAY by Year ---
fig, ax = plt.subplots(figsize=(12, 6))
data_clip = df_operated[['YEAR','ARR_DELAY']].copy()
data_clip['ARR_DELAY'] = data_clip['ARR_DELAY'].clip(-60, 200)
sns.violinplot(data=data_clip, x='YEAR', y='ARR_DELAY', ax=ax, inner='box',
               palette='Set2', cut=0)
ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='15-min threshold')
ax.set_title('ARR_DELAY Distribution by Year (Violin Plot)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../reports/figures/08_delay_violin.png', dpi=150)
plt.show()

# %% [markdown]
# ## 2.4 Cancellation & Diversion Analysis

# %%
# --- Chart 9: Cancellation Rate by Year ---
cancel_year = df.groupby('YEAR').agg(
    total=('CANCELLED','count'),
    cancelled=('CANCELLED','sum'),
    diverted=('DIVERTED','sum')
).reset_index()
cancel_year['cancel_rate'] = cancel_year['cancelled'] / cancel_year['total'] * 100
cancel_year['divert_rate'] = cancel_year['diverted'] / cancel_year['total'] * 100

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(cancel_year))
width = 0.35
ax.bar(x - width/2, cancel_year['cancel_rate'], width, label='Cancelled', color='#e74c3c')
ax.bar(x + width/2, cancel_year['divert_rate'], width, label='Diverted', color='#f39c12')
ax.set_xticks(x)
ax.set_xticklabels(cancel_year['YEAR'])
ax.set_ylabel('Rate (%)')
ax.set_title('Cancellation & Diversion Rate by Year (January)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../reports/figures/09_cancel_divert_rate.png', dpi=150)
plt.show()

# %%
# --- Chart 10: Cancellation by Airport (Top 20) ---
cancel_airport = (df[df['CANCELLED']==1].groupby('ORIGIN').size()
                  .reset_index(name='cancellations')
                  .nlargest(20, 'cancellations'))

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(cancel_airport['ORIGIN'], cancel_airport['cancellations'], color='#e74c3c')
ax.set_xlabel('Number of Cancellations')
ax.set_title('Top 20 Airports by Cancellations (Jan 2021–2025)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/10_cancel_by_airport.png', dpi=150)
plt.show()

# %%
# --- Chart 11: Cancellation Code breakdown ---
if 'CANCELLATION_CODE' in df.columns:
    code_map = {'A': 'Carrier', 'B': 'Weather', 'C': 'NAS', 'D': 'Security'}
    cancel_codes = df[df['CANCELLED']==1]['CANCELLATION_CODE'].map(code_map).value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
    ax.pie(cancel_codes, labels=cancel_codes.index, autopct='%1.1f%%', colors=colors,
           startangle=90, textprops={'fontsize': 12})
    ax.set_title('Cancellation Reasons', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../reports/figures/11_cancel_reasons.png', dpi=150)
    plt.show()

# %% [markdown]
# ## 2.5 Delay Cause Decomposition

# %%
# --- Chart 12: Delay Cause Composition by Year (Stacked Bar) ---
cause_cols = ['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
available_causes = [c for c in cause_cols if c in df_operated.columns]

cause_year = df_operated.groupby('YEAR')[available_causes].sum()

fig, ax = plt.subplots(figsize=(12, 7))
cause_year.plot(kind='bar', stacked=True, ax=ax,
                color=['#e74c3c','#3498db','#f39c12','#2ecc71','#9b59b6'])
ax.set_ylabel('Total Delay Minutes')
ax.set_title('Delay Cause Composition by Year (Jan)', fontsize=14, fontweight='bold')
ax.legend(title='Cause', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('../reports/figures/12_delay_causes_year.png', dpi=150)
plt.show()

# %%
# --- Chart 13: Delay Cause by Carrier (Top 10 delayed) ---
carrier_causes = (df_operated.groupby('OP_CARRIER')[available_causes].mean()
                  .sort_values(available_causes[0], ascending=False).head(10))

fig, ax = plt.subplots(figsize=(14, 8))
carrier_causes.plot(kind='barh', stacked=True, ax=ax,
                    color=['#e74c3c','#3498db','#f39c12','#2ecc71','#9b59b6'])
ax.set_xlabel('Average Delay Minutes')
ax.set_title('Delay Causes by Carrier (Top 10)', fontsize=14, fontweight='bold')
ax.legend(title='Cause')
plt.tight_layout()
plt.savefig('../reports/figures/13_delay_causes_carrier.png', dpi=150)
plt.show()

# %% [markdown]
# ## 2.6 Distance Analysis

# %%
# --- Chart 14: OTP vs Distance Group ---
dist_otp = (df_operated.groupby('DISTANCE_GROUP')
            .agg(flights=('ARR_DEL15','count'), delay_rate=('ARR_DEL15','mean'))
            .reset_index())
dist_otp['OTP'] = 1 - dist_otp['delay_rate']

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.bar(dist_otp['DISTANCE_GROUP'], dist_otp['flights'], alpha=0.3, color='steelblue', label='# Flights')
ax2.plot(dist_otp['DISTANCE_GROUP'], dist_otp['OTP']*100, 'ro-', linewidth=2, label='OTP %')
ax1.set_xlabel('Distance Group')
ax1.set_ylabel('Number of Flights')
ax2.set_ylabel('OTP (%)')
ax1.set_title('OTP vs Distance Group', fontsize=14, fontweight='bold')
fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
plt.tight_layout()
plt.savefig('../reports/figures/14_otp_vs_distance.png', dpi=150)
plt.show()

# %%
# --- Chart 15: Taxi Out vs ARR_DELAY (Hexbin) ---
fig, ax = plt.subplots(figsize=(10, 8))
mask = df_operated['TAXI_OUT'].notna() & df_operated['ARR_DELAY'].notna()
data = df_operated.loc[mask]
hb = ax.hexbin(data['TAXI_OUT'].clip(0, 60), data['ARR_DELAY'].clip(-30, 120),
               gridsize=30, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, label='Count')
ax.set_xlabel('Taxi Out (minutes)')
ax.set_ylabel('ARR_DELAY (minutes)')
ax.set_title('Taxi Out vs Arrival Delay (Hexbin)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/15_taxi_vs_delay.png', dpi=150)
plt.show()

# %% [markdown]
# ## 2.7 Additional Charts

# %%
# --- Chart 16: OTP by Day of Week ---
dow_otp = df_operated.groupby('DAY_OF_WEEK')['ARR_DEL15'].mean()
dow_otp = 1 - dow_otp
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(dow_names, dow_otp.values * 100, color=['#3498db']*5 + ['#e74c3c']*2)
ax.set_ylabel('OTP (%)')
ax.set_title('OTP by Day of Week (Jan 2021–2025)', fontsize=14, fontweight='bold')
for i, v in enumerate(dow_otp.values):
    ax.text(i, v*100 + 0.3, f'{v*100:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/16_otp_by_dow.png', dpi=150)
plt.show()

# %%
# --- Chart 17: OTP Trend over Jan days (line, by year) ---
daily_otp = (df_operated.groupby(['YEAR','DAY_OF_MONTH'])['ARR_DEL15']
             .mean().unstack(level=0))
daily_otp = 1 - daily_otp

fig, ax = plt.subplots(figsize=(14, 6))
for year in daily_otp.columns:
    ax.plot(daily_otp.index, daily_otp[year]*100, 'o-', label=str(year), alpha=0.8)
ax.set_xlabel('Day of January')
ax.set_ylabel('OTP (%)')
ax.set_title('Daily OTP Trend in January', fontsize=14, fontweight='bold')
ax.legend(title='Year')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/17_daily_otp_trend.png', dpi=150)
plt.show()

# %%
# --- Chart 18: Correlation Heatmap (numeric columns) ---
numeric_cols = ['DEP_DELAY','ARR_DELAY','TAXI_OUT','TAXI_IN','DISTANCE',
                'CRS_ELAPSED_TIME','AIR_TIME','CARRIER_DELAY','WEATHER_DELAY',
                'NAS_DELAY','LATE_AIRCRAFT_DELAY']
avail = [c for c in numeric_cols if c in df_operated.columns]

fig, ax = plt.subplots(figsize=(12, 10))
corr = df_operated[avail].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            center=0, square=True, linewidths=0.5)
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/18_correlation_matrix.png', dpi=150)
plt.show()

# %%
# --- Chart 19: Average Delay by Hour of Day ---
df_operated['DEP_HOUR'] = df_operated['CRS_DEP_TIME_MIN'] // 60
hourly = df_operated.groupby('DEP_HOUR')['ARR_DELAY'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hourly.index, hourly.values, 'o-', color='#e74c3c', linewidth=2)
ax.fill_between(hourly.index, hourly.values, alpha=0.2, color='#e74c3c')
ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('Departure Hour')
ax.set_ylabel('Average ARR_DELAY (minutes)')
ax.set_title('Average Arrival Delay by Departure Hour', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig('../reports/figures/19_delay_by_hour.png', dpi=150)
plt.show()

# %%
# --- Chart 20: Delay Category Pie Chart ---
fig, ax = plt.subplots(figsize=(8, 8))
cat_counts = df_operated['DELAY_CAT'].value_counts()
colors = ['#2ecc71','#f1c40f','#e67e22','#e74c3c','#8e44ad']
ax.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', colors=colors,
       startangle=90, textprops={'fontsize': 11})
ax.set_title('Flight Delay Categories', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/20_delay_categories.png', dpi=150)
plt.show()

# %% [markdown]
# ## 2.8 Lưu cleaned data

# %%
# Save operated flights for modeling
df_operated.to_parquet('../data/processed/flights_operated.parquet', index=False)
df.to_parquet('../data/processed/flights_full_cleaned.parquet', index=False)
print(f"✅ Saved operated: {len(df_operated):,} rows")
print(f"✅ Saved full:     {len(df):,} rows")

# %% [markdown]
# ---
# **Next**: `03_feature_engineering.ipynb` → Feature Engineering cho ML
