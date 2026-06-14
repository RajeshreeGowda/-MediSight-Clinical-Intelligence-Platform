CREATE DATABASE medisight_db;
SHOW DATABASES;
USE  medisight_db;
 SHOW TABLES FROM medisight_db;
 SELECT * FROM admissions;
 SELECT * FROM diagnose1;
 SELECT * FROM prescriptions;
 SELECT * FROM patients;
 -- =============================================================
--  MediSight · Analytics Queries (5 Queries)
--  Covers: JOINs, CTEs, Window Functions, CASE, Subqueries
-- =============================================================
USE medisight_db;

-- ─────────────────────────────────────────────────────────────
-- QUERY 1: Readmission Rate by Insurance Type
-- Shows which payer groups have highest readmission burden
-- ─────────────────────────────────────────────────────────────
DESCRIBE admissions;
 

ALTER TABLE admissions
ADD COLUMN readmission_30d TINYINT(1) DEFAULT 0;



UPDATE admissions a1
SET a1.readmission_30d = 1
WHERE EXISTS (
  SELECT 1 FROM admissions a2
  WHERE a2.subject_id  = a1.subject_id
    AND a2.admittime   > a1.dischtime
    AND DATEDIFF(a2.admittime, a1.dischtime) <= 30
);


-- ─────────────────────────────────────────────────────────────
-- QUERY 1: Readmission Rate by Insurance Type
-- Shows which payer groups have highest readmission burden
-- ─────────────────────────────────────────────────────────────
 
SELECT
  insurance,
  COUNT(*) AS total_visits,
  SUM(readmission_30d) AS readmitted,
  COUNT(*) - SUM(readmission_30d) AS not_readmitted,
  ROUND(AVG(readmission_30d) * 100, 1) AS readmit_pct,
  ROUND(AVG(DATEDIFF(DISCHTIME, ADMITTIME)), 1) AS avg_los_days
FROM admissions
WHERE insurance IS NOT NULL
GROUP BY insurance
ORDER BY readmit_pct DESC;


-- ─────────────────────────────────────────────────────────────
-- QUERY 2: Length of Stay CTE + Risk Tier Segmentation
-- Uses CTE + CASE to classify patients into risk tiers
-- ─────────────────────────────────────────────────────────────
WITH los_calc AS (
    SELECT
        a.hadm_id,
        a.subject_id,
        p.gender,
        TIMESTAMPDIFF(YEAR, p.dob, a.admittime)       AS age,
        DATEDIFF(a.dischtime, a.admittime)             AS los_days,
        a.insurance,
        a.discharge_location,
        a.readmission_30d
    FROM admissions a
    JOIN patients p USING(subject_id)
),
risk_tiered AS (
    SELECT *,
        CASE
            WHEN los_days > 7  THEN 'High Risk'
            WHEN los_days > 3  THEN 'Medium Risk'
            ELSE                    'Low Risk'
        END AS risk_tier
    FROM los_calc
)
SELECT
    risk_tier,
    COUNT(*)                                           AS patient_count,
    ROUND(AVG(los_days), 1)                            AS avg_los,
    ROUND(AVG(readmission_30d) * 100, 1)               AS readmit_pct,
    ROUND(AVG(age), 1)                                 AS avg_age
FROM risk_tiered
GROUP BY risk_tier
ORDER BY FIELD(risk_tier, 'High Risk', 'Medium Risk', 'Low Risk');


-- ─────────────────────────────────────────────────────────────
-- QUERY 3: Top 10 ICD-9 Diagnoses by Readmission Count
-- Identifies which conditions drive most readmissions
-- ─────────────────────────────────────────────────────────────
SELECT
    d.icd9_code,
    COUNT(DISTINCT d.hadm_id) AS total_admissions,
    SUM(a.readmission_30d)  AS readmissions,
    ROUND(SUM(a.readmission_30d) /
          COUNT(DISTINCT d.hadm_id) * 100, 1)   AS readmit_pct,
    ROUND(AVG(DATEDIFF(a.dischtime, a.admittime)), 1)  AS avg_los
FROM diagnose1 d
JOIN admissions a USING(hadm_id)
WHERE d.seq_num = 1       -- primary diagnosis only
GROUP BY d.icd9_code
HAVING total_admissions >= 1
ORDER BY readmissions DESC
LIMIT 10;


-- ─────────────────────────────────────────────────────────────
-- QUERY 4: Monthly Admissions Trend (Window Functions)
-- Running total + month-over-month growth rate
-- ─────────────────────────────────────────────────────────────
WITH monthly AS (
    SELECT
        DATE_FORMAT(admittime, '%Y-%m')                AS month,
        COUNT(*)                                       AS admissions,
        SUM(readmission_30d)                           AS readmissions
    FROM admissions
    GROUP BY DATE_FORMAT(admittime, '%Y-%m')
)
SELECT
    month,
    admissions,
    readmissions,
    ROUND(readmissions / admissions * 100, 1)          AS readmit_pct,
    SUM(admissions)   OVER (ORDER BY month)            AS running_total,
    LAG(admissions, 1) OVER (ORDER BY month)           AS prev_month,
    ROUND(
        (admissions - LAG(admissions,1) OVER (ORDER BY month)) /
        NULLIF(LAG(admissions,1) OVER (ORDER BY month), 0) * 100
    , 1)                                               AS mom_growth_pct
FROM monthly
ORDER BY month;


-- ─────────────────────────────────────────────────────────────
-- QUERY 5: Chronic Care Cohort (3+ Visits) + Drug Burden
-- Identifies frequent flyers and their medication complexity
-- ─────────────────────────────────────────────────────────────
WITH visit_counts AS (
    SELECT
        subject_id,
        COUNT(DISTINCT hadm_id)                        AS total_visits,
        SUM(readmission_30d)                           AS total_readmissions,
        ROUND(AVG(DATEDIFF(dischtime, admittime)), 1)  AS avg_los,
        MAX(admittime)                                 AS last_admit
    FROM admissions
    GROUP BY subject_id
    HAVING total_visits >= 3          -- >=2 for demo data, use >=3 with full dataset
),
drug_burden AS (
    SELECT
        subject_id,
        COUNT(DISTINCT drug)                           AS unique_drugs,
        COUNT(*)                                       AS total_rx
    FROM prescriptions
    GROUP BY subject_id
),
diag_burden AS (
    SELECT
        subject_id,
        COUNT(DISTINCT icd9_code)                      AS unique_diagnoses
    FROM diagnose1
    GROUP BY subject_id
)
SELECT
    vc.subject_id,
    p.gender,
    TIMESTAMPDIFF(YEAR, p.dob, vc.last_admit)         AS age,
    vc.total_visits,
    vc.total_readmissions,
    vc.avg_los,
    COALESCE(db.unique_drugs, 0)                       AS unique_drugs,
    COALESCE(dg.unique_diagnoses, 0)                   AS unique_diagnoses,
    CASE
        WHEN vc.total_readmissions >= 2
         AND dg.unique_diagnoses   >= 3  THEN 'Critical'
        WHEN vc.total_readmissions >= 1  THEN 'High'
        ELSE                                  'Moderate'
    END AS chronic_risk_level
FROM visit_counts vc
JOIN patients p USING(subject_id)
LEFT JOIN drug_burden  db USING(subject_id)
LEFT JOIN diag_burden  dg USING(subject_id)
ORDER BY vc.total_readmissions DESC, dg.unique_diagnoses DESC;


-- ─────────────────────────────────────────────────────────────
-- MASTER EXPORT QUERY → save as medisight_master.csv
-- Run: mysql -u root -p medisight_db < 03_queries.sql > /tmp/out.csv
-- Or use Workbench: Results → Export
-- ─────────────────────────────────────────────────────────────
 
SELECT
    p.subject_id,
    p.gender,
    TIMESTAMPDIFF(YEAR, p.dob, a.admittime)            AS age,
    DATEDIFF(a.dischtime, a.admittime)                 AS los_days,
    a.admission_type,
    a.insurance,
    a.discharge_location,
    a.readmission_30d                                  AS target,
    COALESCE(d.num_diagnoses, 0)                       AS num_diagnoses,
    COALESCE(rx.num_drugs, 0)                          AS num_drugs,
    CASE
        WHEN DATEDIFF(a.dischtime, a.admittime) > 7   THEN 'Long'
        WHEN DATEDIFF(a.dischtime, a.admittime) > 3   THEN 'Medium'
        ELSE                                                'Short'
    END                                                AS los_category,
    CASE
        WHEN TIMESTAMPDIFF(YEAR, p.dob, a.admittime) < 30 THEN 'Under 30'
        WHEN TIMESTAMPDIFF(YEAR, p.dob, a.admittime) < 50 THEN '30-50'
        ELSE                                                    'Over 50'
    END                                                AS age_group
FROM patients p
JOIN admissions a
        ON a.subject_id = p.subject_id
LEFT JOIN (
        SELECT subject_id, hadm_id, COUNT(DISTINCT icd9_code) AS num_diagnoses
        FROM diagnose1
        GROUP BY subject_id, hadm_id
     ) d
        ON d.subject_id = a.subject_id
       AND d.hadm_id    = a.hadm_id
LEFT JOIN (
        SELECT subject_id, hadm_id, COUNT(DISTINCT drug) AS num_drugs
        FROM prescriptions
        GROUP BY subject_id, hadm_id
     ) rx
        ON rx.subject_id = a.subject_id
       AND rx.hadm_id    = a.hadm_id
ORDER BY p.subject_id;
