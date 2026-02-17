-- Star schema design for analytics built from student_database_schema.sql
-- Grain choices:
--   fact_watch_events: one row per student-course-day watch event
--   fact_purchases: one row per purchase transaction
--   fact_certificates: one row per certificate issuance
-- Target dialect: MySQL 8+

CREATE DATABASE IF NOT EXISTS data_scientist_dw;
USE data_scientist_dw;

/* =========================
   Dimension tables
   ========================= */

-- Calendar dimension for conformed date analytics across all facts.
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INT PRIMARY KEY,               -- YYYYMMDD surrogate key
    full_date DATE NOT NULL UNIQUE,
    day_of_month TINYINT NOT NULL,
    month_number TINYINT NOT NULL,
    month_name VARCHAR(12) NOT NULL,
    quarter_number TINYINT NOT NULL,
    year_number SMALLINT NOT NULL,
    week_of_year TINYINT NOT NULL,
    is_weekend BOOLEAN NOT NULL
);

-- Student dimension sourced from student_info.
CREATE TABLE IF NOT EXISTS dim_student (
    student_key BIGINT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL UNIQUE,         -- natural key from source
    registration_date_key INT NULL,
    registration_date DATE NULL,
    FOREIGN KEY (registration_date_key) REFERENCES dim_date(date_key)
);

-- Plan dimension sourced from student_purchases.plan_id.
CREATE TABLE IF NOT EXISTS dim_plan (
    plan_key INT AUTO_INCREMENT PRIMARY KEY,
    plan_id TINYINT NOT NULL UNIQUE,
    plan_name VARCHAR(100) NULL,            -- optional enrichment
    billing_period VARCHAR(30) NULL,        -- optional enrichment
    price_usd DECIMAL(10,2) NULL            -- optional enrichment
);

-- Course dimension sourced from student_video_watched.course_id.
CREATE TABLE IF NOT EXISTS dim_course (
    course_key INT AUTO_INCREMENT PRIMARY KEY,
    course_id INT NOT NULL UNIQUE,
    course_name VARCHAR(255) NULL,          -- optional enrichment
    course_track VARCHAR(100) NULL          -- optional enrichment
);

/* =========================
   Fact tables
   ========================= */

-- Watch behavior fact (engagement fact).
CREATE TABLE IF NOT EXISTS fact_watch_events (
    watch_event_key BIGINT AUTO_INCREMENT PRIMARY KEY,
    student_key BIGINT NOT NULL,
    course_key INT NOT NULL,
    watch_date_key INT NOT NULL,
    seconds_watched INT NOT NULL,
    watch_events_count INT NOT NULL DEFAULT 1,
    FOREIGN KEY (student_key) REFERENCES dim_student(student_key),
    FOREIGN KEY (course_key) REFERENCES dim_course(course_key),
    FOREIGN KEY (watch_date_key) REFERENCES dim_date(date_key),
    INDEX idx_fwe_student_date (student_key, watch_date_key),
    INDEX idx_fwe_course_date (course_key, watch_date_key)
);

-- Purchase transaction fact.
CREATE TABLE IF NOT EXISTS fact_purchases (
    purchase_key BIGINT AUTO_INCREMENT PRIMARY KEY,
    purchase_id INT NOT NULL UNIQUE,        -- transaction id from source
    student_key BIGINT NOT NULL,
    plan_key INT NULL,
    purchase_date_key INT NULL,
    refund_date_key INT NULL,
    is_refunded BOOLEAN NOT NULL,
    purchase_count INT NOT NULL DEFAULT 1,
    refund_count INT NOT NULL DEFAULT 0,
    FOREIGN KEY (student_key) REFERENCES dim_student(student_key),
    FOREIGN KEY (plan_key) REFERENCES dim_plan(plan_key),
    FOREIGN KEY (purchase_date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (refund_date_key) REFERENCES dim_date(date_key),
    INDEX idx_fp_student_purchase_date (student_key, purchase_date_key),
    INDEX idx_fp_plan_purchase_date (plan_key, purchase_date_key)
);

-- Certificate issuance fact.
CREATE TABLE IF NOT EXISTS fact_certificates (
    certificate_fact_key BIGINT AUTO_INCREMENT PRIMARY KEY,
    certificate_id INT NOT NULL UNIQUE,
    student_key BIGINT NOT NULL,
    issue_date_key INT NULL,
    certificates_count INT NOT NULL DEFAULT 1,
    FOREIGN KEY (student_key) REFERENCES dim_student(student_key),
    FOREIGN KEY (issue_date_key) REFERENCES dim_date(date_key),
    INDEX idx_fc_student_issue_date (student_key, issue_date_key)
);

/* =========================
   Optional bootstrap/load skeleton
   (kept as reference; run after populating dim_date)
   ========================= */

-- 1) Seed student dimension
-- INSERT INTO dim_student (student_id, registration_date_key, registration_date)
-- SELECT
--     si.student_id,
--     DATE_FORMAT(si.date_registered, '%Y%m%d') + 0 AS registration_date_key,
--     si.date_registered
-- FROM data_scientist_project.student_info si;

-- 2) Seed plan and course dimensions from distinct IDs
-- INSERT INTO dim_plan (plan_id)
-- SELECT DISTINCT sp.plan_id
-- FROM data_scientist_project.student_purchases sp
-- WHERE sp.plan_id IS NOT NULL;

-- INSERT INTO dim_course (course_id)
-- SELECT DISTINCT svw.course_id
-- FROM data_scientist_project.student_video_watched svw
-- WHERE svw.course_id IS NOT NULL;

-- 3) Load purchase fact
-- INSERT INTO fact_purchases (
--     purchase_id, student_key, plan_key, purchase_date_key, refund_date_key,
--     is_refunded, purchase_count, refund_count
-- )
-- SELECT
--     sp.purchase_id,
--     ds.student_key,
--     dp.plan_key,
--     CASE WHEN sp.date_purchased IS NOT NULL THEN DATE_FORMAT(sp.date_purchased, '%Y%m%d') + 0 END,
--     CASE WHEN sp.date_refunded  IS NOT NULL THEN DATE_FORMAT(sp.date_refunded,  '%Y%m%d') + 0 END,
--     (sp.date_refunded IS NOT NULL) AS is_refunded,
--     1,
--     CASE WHEN sp.date_refunded IS NOT NULL THEN 1 ELSE 0 END
-- FROM data_scientist_project.student_purchases sp
-- JOIN dim_student ds ON ds.student_id = sp.student_id
-- LEFT JOIN dim_plan dp ON dp.plan_id = sp.plan_id;

-- 4) Load watch fact
-- INSERT INTO fact_watch_events (student_key, course_key, watch_date_key, seconds_watched, watch_events_count)
-- SELECT
--     ds.student_key,
--     dc.course_key,
--     DATE_FORMAT(svw.date_watched, '%Y%m%d') + 0,
--     COALESCE(svw.seconds_watched, 0),
--     1
-- FROM data_scientist_project.student_video_watched svw
-- JOIN dim_student ds ON ds.student_id = svw.student_id
-- JOIN dim_course dc ON dc.course_id = svw.course_id
-- WHERE svw.date_watched IS NOT NULL;

-- 5) Load certificate fact
-- INSERT INTO fact_certificates (certificate_id, student_key, issue_date_key, certificates_count)
-- SELECT
--     sc.certificate_id,
--     ds.student_key,
--     CASE WHEN sc.date_issued IS NOT NULL THEN DATE_FORMAT(sc.date_issued, '%Y%m%d') + 0 END,
--     1
-- FROM data_scientist_project.student_certificates sc
-- JOIN dim_student ds ON ds.student_id = sc.student_id;
