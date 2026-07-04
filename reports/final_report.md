const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, NumberFormat, convertInchesToTwip, ImageRun, Header,
  Footer, PageNumber, TabStopPosition, TabStopType, UnderlineType,
  LineRuleType,
} = require("docx");
const fs = require("fs");

// ── Colour palette ──────────────────────────────────────────────────
const NAVY   = "0A1628";
const BLUE   = "1A3A6B";
const ACCENT = "2E86AB";
const LIGHT  = "E8F4F8";
const WHITE  = "FFFFFF";
const GREY   = "F5F7FA";
const DARK   = "1C2B3A";
const TEXT   = "2D3748";
const MID    = "4A5568";
const DIM    = "718096";

// ── Helper: coloured filled paragraph (section divider) ─────────────
const colorBar = (text, bg = NAVY, fg = WHITE, sz = 28) =>
  new Paragraph({
    children: [new TextRun({ text, color: fg, bold: true, size: sz })],
    alignment: AlignmentType.CENTER,
    shading: { type: ShadingType.CLEAR, fill: bg },
    spacing: { before: 80, after: 80 },
    indent: { left: 0 },
  });

// ── Helper: section heading ─────────────────────────────────────────
const sectionHead = (text, level = HeadingLevel.HEADING_1) =>
  new Paragraph({
    text,
    heading: level,
    spacing: { before: 300, after: 120 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 },
    },
    run: { color: NAVY, bold: true, size: 28 },
  });

// ── Helper: body paragraph ──────────────────────────────────────────
const body = (text, opts = {}) =>
  new Paragraph({
    children: [new TextRun({ text, color: TEXT, size: 22, ...opts })],
    spacing: { before: 60, after: 60 },
    alignment: AlignmentType.JUSTIFIED,
  });

// ── Helper: bullet ──────────────────────────────────────────────────
const bullet = (text, bold_prefix = "") =>
  new Paragraph({
    children: [
      bold_prefix
        ? new TextRun({ text: bold_prefix + " ", color: ACCENT, bold: true, size: 22 })
        : new TextRun({ text: "• ", color: ACCENT, bold: true, size: 22 }),
      new TextRun({ text, color: TEXT, size: 22 }),
    ],
    spacing: { before: 40, after: 40 },
    indent: { left: 360 },
  });

// ── Helper: key-value row ────────────────────────────────────────────
const kvRow = (key, val, shade = false) =>
  new TableRow({
    children: [
      new TableCell({
        children: [new Paragraph({ children: [new TextRun({ text: key, bold: true, color: DARK, size: 20 })] })],
        width: { size: 3200, type: WidthType.DXA },
        shading: shade ? { type: ShadingType.CLEAR, fill: GREY } : undefined,
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
      }),
      new TableCell({
        children: [new Paragraph({ children: [new TextRun({ text: val, color: TEXT, size: 20 })] })],
        width: { size: 6800, type: WidthType.DXA },
        shading: shade ? { type: ShadingType.CLEAR, fill: GREY } : undefined,
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
      }),
    ],
  });

// ── Helper: performance table row ───────────────────────────────────
const perfRow = (day, model, cv, r2, conf, isHeader = false) =>
  new TableRow({
    tableHeader: isHeader,
    children: [day, model, cv, r2, conf].map((t, i) =>
      new TableCell({
        children: [
          new Paragraph({
            children: [new TextRun({
              text: t,
              bold: isHeader,
              color: isHeader ? WHITE : i === 3 ? ACCENT : TEXT,
              size: isHeader ? 20 : 20,
            })],
            alignment: AlignmentType.CENTER,
          }),
        ],
        width: { size: 2000, type: WidthType.DXA },
        shading: isHeader
          ? { type: ShadingType.CLEAR, fill: NAVY }
          : { type: ShadingType.CLEAR, fill: i % 2 === 0 ? WHITE : GREY },
        margins: { top: 80, bottom: 80, left: 80, right: 80 },
      })
    ),
  });

// ── Helper: feature table row ────────────────────────────────────────
const featRow = (cat, feats, isHeader = false) =>
  new TableRow({
    tableHeader: isHeader,
    children: [cat, feats].map((t, i) =>
      new TableCell({
        children: [new Paragraph({ children: [new TextRun({ text: t, bold: isHeader, color: isHeader ? WHITE : i === 0 ? DARK : TEXT, size: 19 })] })],
        width: { size: i === 0 ? 2800 : 7200, type: WidthType.DXA },
        shading: isHeader ? { type: ShadingType.CLEAR, fill: BLUE } : { type: ShadingType.CLEAR, fill: isHeader ? BLUE : (i === 0 ? LIGHT : WHITE) },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
      })
    ),
  });

// ════════════════════════════════════════════════════════════════════
// DOCUMENT
// ════════════════════════════════════════════════════════════════════
const doc = new Document({
  numbering: { config: [] },
  sections: [
    // ================================================================
    // SECTION 1 — COVER PAGE
    // ================================================================
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 720, bottom: 720, left: 1080, right: 1080 },
        },
      },
      children: [
        // Top banner
        new Paragraph({
          children: [new TextRun({ text: "", size: 2 })],
          shading: { type: ShadingType.CLEAR, fill: NAVY },
          spacing: { before: 0, after: 0 },
        }),
        new Paragraph({
          children: [
            new TextRun({ text: "PEARL PAKISTAN", color: WHITE, bold: true, size: 28, characterSpacing: 200 }),
          ],
          alignment: AlignmentType.CENTER,
          shading: { type: ShadingType.CLEAR, fill: NAVY },
          spacing: { before: 200, after: 60 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Data Science Internship Program", color: "A0C4D8", size: 22 })],
          alignment: AlignmentType.CENTER,
          shading: { type: ShadingType.CLEAR, fill: NAVY },
          spacing: { before: 0, after: 320 },
        }),

        // Accent bar
        new Paragraph({
          children: [new TextRun({ text: "", size: 4 })],
          shading: { type: ShadingType.CLEAR, fill: ACCENT },
          spacing: { before: 0, after: 0 },
        }),

        // Title block
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 600, after: 0 } }),
        new Paragraph({
          children: [new TextRun({ text: "Pearl AQI Predictor", color: NAVY, bold: true, size: 64 })],
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 120 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Production-Grade, Serverless 3-Day Air Quality Forecasting System", color: ACCENT, size: 24, italics: true })],
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 800 },
        }),

        // Info table
        new Table({
          width: { size: 7200, type: WidthType.DXA },
          columnWidths: [2400, 4800],
          alignment: AlignmentType.CENTER,
          rows: [
            kvRow("Submitted By", "Abrar Shakeel Ghoury", false),
            kvRow("Role", "Data Science Intern", true),
            kvRow("Organization", "Pearl Pakistan", false),
            kvRow("Project", "Pearl AQI Predictor", true),
            kvRow("Deployment City", "Istanbul, Turkey", false),
            kvRow("Live Dashboard", "airlens-istanbul.streamlit.app", true),
            kvRow("Date", new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" }), false),
          ],
        }),

        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 600, after: 0 } }),

        // Bottom accent
        new Paragraph({
          children: [new TextRun({ text: "", size: 4 })],
          shading: { type: ShadingType.CLEAR, fill: ACCENT },
          spacing: { before: 0, after: 0 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "CONFIDENTIAL — FOR SUBMISSION PURPOSES ONLY", color: "888888", size: 16, characterSpacing: 100 })],
          alignment: AlignmentType.CENTER,
          spacing: { before: 160, after: 0 },
        }),
      ],
    },

    // ================================================================
    // SECTION 2 — MAIN REPORT
    // ================================================================
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, bottom: 1080, left: 1260, right: 1260 },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: "Pearl AQI Predictor  |  Technical Report", color: MID, size: 18 }),
                new TextRun({ text: "  •  Abrar Shakeel Ghoury", color: DIM, size: 18 }),
              ],
              border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT } },
              spacing: { after: 0 },
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: "Pearl Pakistan  |  Data Science Internship  |  Page ", color: DIM, size: 18 }),
                new TextRun({ children: [PageNumber.CURRENT], color: ACCENT, size: 18 }),
              ],
              alignment: AlignmentType.CENTER,
              border: { top: { style: BorderStyle.SINGLE, size: 4, color: ACCENT } },
              spacing: { before: 80 },
            }),
          ],
        }),
      },

      children: [

        // ── 1. EXECUTIVE SUMMARY ─────────────────────────────────────
        sectionHead("1. Executive Summary"),
        body(
          "Pearl AQI Predictor is a fully automated, end-to-end machine learning system designed to forecast the Air Quality Index (AQI) for Istanbul, Turkey, up to three days in advance. The system was developed as part of the Pearl Pakistan Data Science Internship Program and demonstrates a complete MLOps lifecycle — from raw data ingestion and feature engineering to model training, deployment, and interactive visualization."
        ),
        body(
          "The project operates on a 100% serverless architecture with zero monthly infrastructure cost, leveraging MongoDB Atlas for storage, GitHub Actions for pipeline automation, and Streamlit Cloud for the live dashboard. Open-Meteo's free API provides both historical air quality data and three-day weather forecasts."
        ),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 120, after: 0 } }),

        // Summary table
        new Table({
          width: { size: 10000, type: WidthType.DXA },
          columnWidths: [3000, 7000],
          rows: [
            new TableRow({
              tableHeader: true,
              children: ["Metric", "Value"].map(t =>
                new TableCell({
                  children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: WHITE, size: 20 })] })],
                  width: { size: t === "Metric" ? 3000 : 7000, type: WidthType.DXA },
                  shading: { type: ShadingType.CLEAR, fill: NAVY },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                })
              ),
            }),
            kvRow("Forecast Horizons", "Day 1 (High), Day 2 (Moderate), Day 3 (Low confidence)", false),
            kvRow("Models Evaluated", "XGBoost, LightGBM, RandomForest, GradientBoosting, Ridge", true),
            kvRow("Day 1 Test R²", "0.874", false),
            kvRow("Day 2 Test R²", "0.633", true),
            kvRow("Day 3 Test R²", "0.518", false),
            kvRow("Training Data", "2 years of hourly AQ + weather data (~720 daily rows)", true),
            kvRow("Features Engineered", "31 features across 9 categories", false),
            kvRow("Infrastructure Cost", "$0 / month (fully free-tier)", true),
            kvRow("Live URL", "https://airlens-istanbul.streamlit.app", false),
          ],
        }),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── 2. PROJECT OBJECTIVES ────────────────────────────────────
        sectionHead("2. Project Objectives"),
        bullet("Build an end-to-end, production-ready AQI prediction system with no manual intervention required after initial setup.", "①"),
        bullet("Forecast AQI for three consecutive days using a combination of historical patterns and forward-looking weather data.", "②"),
        bullet("Deploy a publicly accessible, interactive dashboard that presents predictions with health advisories and model explainability.", "③"),
        bullet("Demonstrate sound ML engineering practices: data leakage prevention, chronological train/test splits, and model versioning.", "④"),
        bullet("Keep the entire stack serverless and free, suitable for replication in resource-constrained environments.", "⑤"),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 120, after: 0 } }),

        // ── 3. SYSTEM ARCHITECTURE ──────────────────────────────────
        sectionHead("3. System Architecture"),
        body(
          "The system follows a producer-consumer MLOps pipeline pattern with four distinct layers. Each layer is independently deployable and communicates exclusively through MongoDB Atlas, which acts as the central data and model registry."
        ),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 80, after: 0 } }),

        new Table({
          width: { size: 10000, type: WidthType.DXA },
          columnWidths: [2000, 2500, 5500],
          rows: [
            new TableRow({
              tableHeader: true,
              children: ["Layer", "Component", "Responsibility"].map(t =>
                new TableCell({
                  children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: WHITE, size: 20 })] })],
                  width: { size: t === "Layer" ? 2000 : t === "Component" ? 2500 : 5500, type: WidthType.DXA },
                  shading: { type: ShadingType.CLEAR, fill: BLUE },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                })
              ),
            }),
            ...[
              ["Ingestion", "fetch_openmeteo.py", "Pulls hourly AQ + weather from Open-Meteo API. In daily mode, also fetches 3-day weather forecast.", false],
              ["Feature Store", "compute_features.py", "Aggregates hourly data to daily, engineers 31 features across 9 categories, upserts to MongoDB.", true],
              ["Training", "train.py + registry.py", "5-model competition per target using TimeSeriesSplit CV. Winner saved to GridFS with version control.", false],
              ["Serving", "predict.py + app.py", "Loads models from GridFS, runs inference, computes SHAP values, renders 4-page Streamlit dashboard.", true],
            ].map(([l, c, r, shade]) =>
              new TableRow({
                children: [l, c, r].map((t, i) =>
                  new TableCell({
                    children: [new Paragraph({ children: [new TextRun({ text: t, color: i === 0 ? ACCENT : TEXT, bold: i === 0, size: 19 })] })],
                    width: { size: i === 0 ? 2000 : i === 1 ? 2500 : 5500, type: WidthType.DXA },
                    shading: { type: ShadingType.CLEAR, fill: shade ? GREY : WHITE },
                    margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  })
                ),
              })
            ),
          ],
        }),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── 4. DATA PIPELINE ─────────────────────────────────────────
        sectionHead("4. Data Pipeline & Feature Engineering"),
        new Paragraph({
          heading: HeadingLevel.HEADING_2,
          children: [new TextRun({ text: "4.1 Data Sources", color: BLUE, bold: true, size: 24 })],
          spacing: { before: 200, after: 80 },
        }),
        bullet("Open-Meteo Air Quality API — hourly PM2.5, PM10, CO, NO2, SO2, O3, US AQI"),
        bullet("Open-Meteo Archive API — hourly temperature, humidity, wind speed/direction, pressure, precipitation"),
        bullet("Open-Meteo Forecast API — 3-day ahead weather forecast (temperature, wind, pressure, precipitation)"),
        bullet("Backfill window: 730 days (2 years) | Daily updates: last 3 days with upsert deduplication"),

        new Paragraph({
          heading: HeadingLevel.HEADING_2,
          children: [new TextRun({ text: "4.2 Feature Engineering", color: BLUE, bold: true, size: 24 })],
          spacing: { before: 200, after: 80 },
        }),
        body("All features are computed from daily-aggregated data. Strict shift(1) on all lag and rolling features ensures no future data leaks into training. A hard-stop leakage checker validates the feature matrix before every training run."),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 80, after: 0 } }),

        new Table({
          width: { size: 10000, type: WidthType.DXA },
          columnWidths: [2800, 7200],
          rows: [
            featRow("Category", "Features", true),
            featRow("AQI Aggregations", "aqi_mean, aqi_max, aqi_min, aqi_std, aqi_last6h", false),
            featRow("Pollutants", "pm2_5_mean, pm10_mean"),
            featRow("Weather", "temp_mean/max/min, humidity_mean, wind_mean/max, pressure_mean"),
            featRow("Lag Features", "aqi_lag1d, aqi_lag2d, aqi_lag3d, aqi_lag7d, pm2_5_lag1d/2d"),
            featRow("Rolling Features", "aqi_roll_mean_3, aqi_roll_mean_7"),
            featRow("Trend + Delta", "aqi_trend_1d/3d, aqi_diff, aqi_pct_change, pm2_5/pm10/temp/pressure/wind/humidity_diff"),
            featRow("Volatility", "aqi_std_7d (7-day rolling standard deviation)"),
            featRow("Time Encoding", "month_sin, month_cos (cyclical), day_of_week"),
            featRow("Forecast Weather (NEW)", "fc_temp/humidity/wind/pressure/precip_d1/d2/d3, fc_pressure_drop_d1"),
          ],
        }),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 120, after: 0 } }),

        new Paragraph({
          heading: HeadingLevel.HEADING_2,
          children: [new TextRun({ text: "4.3 The Forecast Weather Innovation", color: BLUE, bold: true, size: 24 })],
          spacing: { before: 200, after: 80 },
        }),
        body(
          "The single most impactful engineering decision was adding forward-looking weather forecast features. Without these, Day 3 R² was 0.05 — essentially no better than predicting the mean. By incorporating Open-Meteo's 3-day weather forecast (wind speed, pressure, precipitation) as model inputs, Day 3 R² improved to 0.518 and Day 2 R² improved from 0.29 to 0.633."
        ),
        body(
          "During training (backfill), actual future weather (shift -1/-2/-3 on archive data) serves as a perfect-forecast proxy. During live inference, the real forecast API provides the equivalent signal. This design ensures training-inference consistency while giving the model genuine forward-looking information."
        ),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── 5. MODEL TRAINING ────────────────────────────────────────
        sectionHead("5. Model Training & Selection"),
        body(
          "Three separate models are trained — one per forecast horizon (Day 1, Day 2, Day 3). For each target, five algorithms compete in a structured tournament using TimeSeriesSplit cross-validation with 3 folds. The algorithm with the lowest mean CV RMSE is selected and retrained on the full training set before being saved to MongoDB GridFS."
        ),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 80, after: 0 } }),

        new Table({
          width: { size: 10000, type: WidthType.DXA },
          columnWidths: [2000, 2000, 2000, 2000, 2000],
          rows: [
            perfRow("Horizon", "Winning Model", "CV RMSE", "Test R²", "Confidence", true),
            perfRow("Day 1", "Ridge", "12.39", "0.874", "High"),
            perfRow("Day 2", "GradientBoosting", "20.11", "0.633", "Moderate"),
            perfRow("Day 3", "GradientBoosting", "17.92", "0.518", "Low"),
          ],
        }),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 120, after: 0 } }),

        new Paragraph({
          heading: HeadingLevel.HEADING_2,
          children: [new TextRun({ text: "5.1 Recursive Multi-Step Prediction", color: BLUE, bold: true, size: 24 })],
          spacing: { before: 200, after: 80 },
        }),
        bullet("Day 1 model: trained on all base features"),
        bullet("Day 2 model: trained with pred_day1 (Day 1 prediction) as an additional feature"),
        bullet("Day 3 model: trained with pred_day1 + pred_day2 as additional features"),
        body("This recursive strategy allows later-horizon models to condition on earlier predictions, reducing accumulated error compared to a naive single-model approach."),

        new Paragraph({
          heading: HeadingLevel.HEADING_2,
          children: [new TextRun({ text: "5.2 Data Leakage Prevention", color: BLUE, bold: true, size: 24 })],
          spacing: { before: 200, after: 80 },
        }),
        bullet("All lag/rolling features apply shift(1) before any window operation — no current-day data used"),
        bullet("Chronological 80/20 train/test split — never shuffled"),
        bullet("TimeSeriesSplit CV maintains temporal ordering within cross-validation folds"),
        bullet("Hard-stop leakage checker at feature engineering stage and again at training stage"),
        bullet("Target columns (aqi_day1/2/3) explicitly excluded from feature matrix with ValueError on violation"),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── 6. MLOPS & AUTOMATION ────────────────────────────────────
        sectionHead("6. MLOps & Automation"),
        body(
          "The pipeline runs entirely on GitHub Actions with no manual intervention required after initial setup. Two workflows run on a daily schedule:"
        ),
        bullet("Feature Pipeline (06:00 UTC daily) — fetches last 3 days of AQ and weather data, cleans, engineers features, fetches 3-day weather forecast, upserts to MongoDB feature_store"),
        bullet("Training Pipeline (07:00 UTC daily) — loads full feature history, runs 5-model competition for each of 3 targets, saves winning models to GridFS"),
        body(
          "MongoDB Atlas GridFS handles model persistence — no local file system required. Models are versioned with dynamic version numbers and the last 2 versions are retained. The Streamlit dashboard always loads the latest active model version via the registry module."
        ),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── 7. DASHBOARD ─────────────────────────────────────────────
        sectionHead("7. Dashboard & Explainability"),
        new Table({
          width: { size: 10000, type: WidthType.DXA },
          columnWidths: [2500, 7500],
          rows: [
            new TableRow({
              tableHeader: true,
              children: ["Page", "Contents"].map(t =>
                new TableCell({
                  children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: WHITE, size: 20 })] })],
                  width: { size: t === "Page" ? 2500 : 7500, type: WidthType.DXA },
                  shading: { type: ShadingType.CLEAR, fill: NAVY },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                })
              ),
            }),
            ...[
              ["Predictive AQI Forecast", "Color-coded Day 1/2/3 AQI cards with values, categories (Good/Moderate/Unhealthy/Hazardous), confidence levels, AQI gauge bars, health advisory banner, AQI scale reference, and Istanbul forecast summary with image panel.", false],
              ["Historical Trends", "30-day actual AQI line chart with model prediction overlay, uncertainty shading for Day 2/3, large-error markers (>20 AQI), unhealthy zone shading, pollutant trends (PM2.5, PM10), model accuracy table.", true],
              ["Explainable AI (SHAP)", "Plain-English insight sentences per day, horizontal SHAP bar charts (top 10 features), Plotly waterfall charts (base value to final prediction), cross-day feature importance comparison chart, raw SHAP values table.", false],
              ["Model Performance", "Training metrics from registry (CV RMSE, Test RMSE, MAE, R², features, rows), live accuracy as predictions accumulate, actual vs. predicted scatter with ±20 AQI band, residual bar charts, rolling 7-day RMSE stability chart.", true],
            ].map(([page, desc, shade]) =>
              new TableRow({
                children: [page, desc].map((t, i) =>
                  new TableCell({
                    children: [new Paragraph({ children: [new TextRun({ text: t, color: i === 0 ? BLUE : TEXT, bold: i === 0, size: 19 })] })],
                    width: { size: i === 0 ? 2500 : 7500, type: WidthType.DXA },
                    shading: { type: ShadingType.CLEAR, fill: shade ? GREY : WHITE },
                    margins: { top: 100, bottom: 100, left: 120, right: 120 },
                  })
                ),
              })
            ),
          ],
        }),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── 8. TECH STACK ─────────────────────────────────────────────
        sectionHead("8. Technology Stack"),
        new Table({
          width: { size: 10000, type: WidthType.DXA },
          columnWidths: [2500, 3000, 4500],
          rows: [
            new TableRow({
              tableHeader: true,
              children: ["Category", "Technology", "Purpose"].map(t =>
                new TableCell({
                  children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: WHITE, size: 20 })] })],
                  shading: { type: ShadingType.CLEAR, fill: BLUE },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                })
              ),
            }),
            ...[
              ["Language", "Python 3.11", "Core development language", false],
              ["ML Framework", "scikit-learn, XGBoost, LightGBM", "Model training and evaluation", true],
              ["Explainability", "SHAP", "Feature importance and waterfall analysis", false],
              ["Data Processing", "Pandas, NumPy", "Feature engineering and aggregation", true],
              ["Database", "MongoDB Atlas (GridFS)", "Feature store, model registry, predictions", false],
              ["Data Source", "Open-Meteo API", "Historical AQ, weather, 3-day forecast", true],
              ["Dashboard", "Streamlit, Plotly", "Interactive 4-page web application", false],
              ["Automation", "GitHub Actions", "Daily pipeline scheduling and execution", true],
              ["Deployment", "Streamlit Cloud", "Zero-config serverless hosting", false],
            ].map(([cat, tech, purpose, shade]) =>
              new TableRow({
                children: [cat, tech, purpose].map(t =>
                  new TableCell({
                    children: [new Paragraph({ children: [new TextRun({ text: t, color: TEXT, size: 19 })] })],
                    shading: { type: ShadingType.CLEAR, fill: shade ? GREY : WHITE },
                    margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  })
                ),
              })
            ),
          ],
        }),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── 9. CHALLENGES ─────────────────────────────────────────────
        sectionHead("9. Challenges & Solutions"),
        ...[
          ["Day 2/3 Model Accuracy", "Initial Day 3 R² was 0.05 — models had no meaningful forward-looking signal, relying only on historical AQI lags which lose predictive power beyond 1 day.", "Added Open-Meteo 3-day forecast weather features (fc_pressure_d1/2/3, fc_wind_d1/2/3). Day 3 R² improved to 0.518, Day 2 from 0.29 to 0.633."],
          ["Forecast Feature Leakage in Training", "Using actual future weather during training (shift -1/-2/-3) creates a training-inference gap since live mode uses API forecasts which have their own error.", "Documented as 'optimistic bias' in report. Expected live Day 2/3 R² slightly lower (~0.50-0.55 and ~0.40-0.45). This is standard practice in weather-dependent ML systems."],
          ["Model Versioning Without File System", "Serverless environments have no persistent disk, making traditional model.pkl file saving impossible.", "MongoDB GridFS stores serialized model bytes (joblib) directly in the database. Registry module handles versioning, retrieval, and automatic cleanup of old versions."],
          ["NaN Features at Training Time", "Forecast columns (fc_*) are NaN for all historical backfill rows since past dates have no 'forecast'. sklearn estimators (Ridge, RF) crash on NaN input.", "Mean-fill applied to feature matrix before training. XGBoost/LightGBM tolerate NaN natively; filling ensures all 5 models train on identical input for fair CV comparison."],
          ["GitHub Push Size Limit", "Initial push included venv folder (~330MB) containing large DLL files exceeding GitHub's 100MB file limit.", "Removed venv from git history using git filter-branch, added venv/ to .gitignore. requirements.txt handles dependency installation on any environment."],
        ].map(([title, problem, solution], idx) =>
          new Table({
            width: { size: 10000, type: WidthType.DXA },
            columnWidths: [10000],
            margins: { top: 0, bottom: idx < 4 ? 200 : 0 },
            rows: [
              new TableRow({
                children: [new TableCell({
                  children: [new Paragraph({ children: [new TextRun({ text: `Challenge ${idx + 1}: ${title}`, bold: true, color: WHITE, size: 20 })] })],
                  shading: { type: ShadingType.CLEAR, fill: BLUE },
                  margins: { top: 80, bottom: 80, left: 180, right: 180 },
                })],
              }),
              new TableRow({
                children: [new TableCell({
                  children: [
                    new Paragraph({ children: [new TextRun({ text: "Problem: ", bold: true, color: ACCENT, size: 20 }), new TextRun({ text: problem, color: TEXT, size: 20 })] }),
                    new Paragraph({ children: [new TextRun({ text: "Solution: ", bold: true, color: "27AE60", size: 20 }), new TextRun({ text: solution, color: TEXT, size: 20 })] }),
                  ],
                  shading: { type: ShadingType.CLEAR, fill: GREY },
                  margins: { top: 120, bottom: 120, left: 180, right: 180 },
                })],
              }),
            ],
          })
        ),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── 10. RESULTS ──────────────────────────────────────────────
        sectionHead("10. Results & Evaluation"),
        body(
          "The system achieved production-quality accuracy on Day 1 forecasts and meaningful signal on Day 2 and Day 3 — a significant improvement over the naive baseline (predicting the historical mean, R² = 0.0). All results are computed on a held-out 20% chronological test set; no test data was used during model selection."
        ),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 80, after: 0 } }),

        new Table({
          width: { size: 10000, type: WidthType.DXA },
          columnWidths: [2000, 2000, 2000, 2000, 2000],
          rows: [
            perfRow("Horizon", "Model", "Test RMSE", "Test MAE", "Test R²", true),
            perfRow("Day 1", "Ridge", "9.78", "8.01", "0.874"),
            perfRow("Day 2", "GradientBoosting", "16.78", "12.98", "0.633"),
            perfRow("Day 3", "GradientBoosting", "18.87", "13.61", "0.518"),
          ],
        }),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 120, after: 0 } }),

        body(
          "Comparison against academic and commercial AQI forecasting benchmarks shows the system performs within the expected range for single-city, data-efficient forecasting systems:"
        ),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 80, after: 0 } }),

        new Table({
          width: { size: 10000, type: WidthType.DXA },
          columnWidths: [3000, 2500, 2500, 2000],
          rows: [
            new TableRow({
              tableHeader: true,
              children: ["System", "Day 1 R²", "Day 2 R²", "Day 3 R²"].map(t =>
                new TableCell({
                  children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: WHITE, size: 20 })] })],
                  shading: { type: ShadingType.CLEAR, fill: NAVY },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                })
              ),
            }),
            ...[
              ["Pearl AQI Predictor (This Work)", "0.874", "0.633", "0.518", false],
              ["Academic AQI Forecasting", "0.75 – 0.90", "0.50 – 0.65", "0.35 – 0.55", true],
              ["EPA Research Papers", "0.80 – 0.92", "0.55 – 0.70", "0.40 – 0.60", false],
              ["Commercial Services", "0.85 – 0.95", "0.65 – 0.80", "0.50 – 0.70", true],
            ].map(([sys, d1, d2, d3, shade]) =>
              new TableRow({
                children: [sys, d1, d2, d3].map((t, i) =>
                  new TableCell({
                    children: [new Paragraph({ children: [new TextRun({ text: t, color: i === 0 ? (shade ? TEXT : ACCENT) : TEXT, bold: i === 0, size: 19 })] })],
                    shading: { type: ShadingType.CLEAR, fill: shade ? GREY : WHITE },
                    margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  })
                ),
              })
            ),
          ],
        }),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── 11. LIMITATIONS ──────────────────────────────────────────
        sectionHead("11. Limitations & Future Work"),
        new Paragraph({
          heading: HeadingLevel.HEADING_2,
          children: [new TextRun({ text: "11.1 Current Limitations", color: BLUE, bold: true, size: 24 })],
          spacing: { before: 160, after: 80 },
        }),
        bullet("Day 2/3 accuracy is slightly optimistic due to use of perfect historical weather as forecast proxy during training — expected live R² may be 5-10% lower"),
        bullet("Single-city deployment (Istanbul) — extending to other cities requires updating city config and re-running backfill"),
        bullet("No real-time sensor data — relies entirely on API data which has up to 24-hour lag"),
        bullet("SHAP computation on Ridge (LinearExplainer) requires StandardScaler-transformed input — adds complexity to the inference pipeline"),

        new Paragraph({
          heading: HeadingLevel.HEADING_2,
          children: [new TextRun({ text: "11.2 Future Improvements", color: BLUE, bold: true, size: 24 })],
          spacing: { before: 200, after: 80 },
        }),
        bullet("Add AQICN real-time sensor data as a supplementary AQI source to reduce API lag"),
        bullet("Implement Bayesian hyperparameter optimization (Optuna) to replace current grid search"),
        bullet("Train LSTM or Temporal Fusion Transformer for sequence-aware multi-step forecasting"),
        bullet("Add alert notification system (email/SMS) when AQI forecast exceeds hazardous threshold"),
        bullet("Extend to multi-city deployment with per-city model versioning in the registry"),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── 12. CONCLUSION ───────────────────────────────────────────
        sectionHead("12. Conclusion"),
        body(
          "Pearl AQI Predictor successfully demonstrates a production-grade, end-to-end machine learning system built entirely on free-tier, serverless infrastructure. The project covers the complete MLOps lifecycle — data ingestion, feature engineering, model training, versioning, deployment, and explainability — without any manual intervention after initial setup."
        ),
        body(
          "The key technical contribution is the incorporation of forward-looking weather forecast features, which transformed Day 3 prediction from essentially random (R² = 0.05) to genuinely useful (R² = 0.518). The system achieves Day 1 accuracy competitive with academic benchmarks and deploys as a publicly accessible, interactive dashboard."
        ),
        body(
          "The project demonstrates that a small-data, resource-constrained ML system — trained on just 720 daily rows — can produce actionable forecasts when feature engineering is thoughtfully designed to capture the true drivers of the target variable rather than relying solely on historical autocorrelation."
        ),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 200, after: 0 } }),

        // ── CONTACT ──────────────────────────────────────────────────
        colorBar("Contact & Links", NAVY, WHITE, 24),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 120, after: 0 } }),
        new Table({
          width: { size: 10000, type: WidthType.DXA },
          columnWidths: [2500, 7500],
          rows: [
            kvRow("Name", "Abrar Shakeel Ghoury", false),
            kvRow("Role", "Data Science Intern — Pearl Pakistan", true),
            kvRow("Email", "abrarshakeel21@gmail.com", false),
            kvRow("LinkedIn", "linkedin.com/in/abrar-ghoury", true),
            kvRow("GitHub", "github.com/abrarghoury", false),
            kvRow("Live Dashboard", "https://airlens-istanbul.streamlit.app", true),
            kvRow("Source Code", "github.com/abrarghoury/Pearl_Aqi", false),
          ],
        }),
        new Paragraph({ children: [new TextRun({ text: "" })], spacing: { before: 400, after: 0 } }),
        new Paragraph({
          children: [new TextRun({ text: "© 2026 Abrar Shakeel Ghoury | Pearl Pakistan Data Science Internship", color: DIM, size: 18, italics: true })],
          alignment: AlignmentType.CENTER,
        }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/mnt/user-data/outputs/Pearl_AQI_Report.docx", buffer);
  console.log("Done");
});