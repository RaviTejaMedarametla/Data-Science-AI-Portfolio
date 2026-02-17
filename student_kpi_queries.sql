-- KPI queries for the EdTech schema defined in student_database_schema.sql
-- Dialect: MySQL 8+

/* ==========================================================
   1) CONVERSION FUNNEL
   Stages:
   - registered_students: distinct students in student_info
   - engaged_students: registered students with any video watch event
   - purchasers: registered students with at least one non-refunded purchase
   - certified_students: registered students with at least one certificate
   ========================================================== */
WITH registered AS (
    SELECT DISTINCT si.student_id
    FROM student_info si
),
engaged AS (
    SELECT DISTINCT svw.student_id
    FROM student_video_watched svw
    WHERE svw.student_id IS NOT NULL
),
purchasers AS (
    SELECT DISTINCT sp.student_id
    FROM student_purchases sp
    WHERE sp.student_id IS NOT NULL
      AND (sp.date_refunded IS NULL OR sp.date_refunded < sp.date_purchased)
),
certified AS (
    SELECT DISTINCT sc.student_id
    FROM student_certificates sc
)
SELECT
    stage,
    users,
    ROUND(users / NULLIF(MAX(users) OVER (), 0), 4) AS conversion_from_registered
FROM (
    SELECT 'registered_students' AS stage, COUNT(*) AS users FROM registered
    UNION ALL
    SELECT 'engaged_students', COUNT(*) FROM engaged e JOIN registered r ON e.student_id = r.student_id
    UNION ALL
    SELECT 'purchasers', COUNT(*) FROM purchasers p JOIN registered r ON p.student_id = r.student_id
    UNION ALL
    SELECT 'certified_students', COUNT(*) FROM certified c JOIN registered r ON c.student_id = r.student_id
) f
ORDER BY FIELD(stage, 'registered_students', 'engaged_students', 'purchasers', 'certified_students');


/* ==========================================================
   2) COHORT RETENTION (monthly)
   Cohort month = month of date_registered
   Retention month = month difference between activity month and cohort month
   Activity source = student_video_watched events
   ========================================================== */
WITH base AS (
    SELECT
        si.student_id,
        DATE_FORMAT(si.date_registered, '%Y-%m-01') AS cohort_month
    FROM student_info si
    WHERE si.date_registered IS NOT NULL
),
activity AS (
    SELECT DISTINCT
        svw.student_id,
        DATE_FORMAT(svw.date_watched, '%Y-%m-01') AS activity_month
    FROM student_video_watched svw
    WHERE svw.student_id IS NOT NULL
      AND svw.date_watched IS NOT NULL
),
cohort_activity AS (
    SELECT
        b.cohort_month,
        TIMESTAMPDIFF(
            MONTH,
            STR_TO_DATE(b.cohort_month, '%Y-%m-01'),
            STR_TO_DATE(a.activity_month, '%Y-%m-01')
        ) AS month_number,
        b.student_id
    FROM base b
    JOIN activity a
      ON a.student_id = b.student_id
     AND STR_TO_DATE(a.activity_month, '%Y-%m-01') >= STR_TO_DATE(b.cohort_month, '%Y-%m-01')
),
cohort_size AS (
    SELECT cohort_month, COUNT(*) AS cohort_users
    FROM base
    GROUP BY cohort_month
)
SELECT
    ca.cohort_month,
    ca.month_number,
    COUNT(DISTINCT ca.student_id) AS retained_users,
    cs.cohort_users,
    ROUND(COUNT(DISTINCT ca.student_id) / NULLIF(cs.cohort_users, 0), 4) AS retention_rate
FROM cohort_activity ca
JOIN cohort_size cs USING (cohort_month)
GROUP BY ca.cohort_month, ca.month_number, cs.cohort_users
ORDER BY ca.cohort_month, ca.month_number;


/* ==========================================================
   3) TIME-TO-PURCHASE
   Calculates days from registration to first valid purchase.
   ========================================================== */
WITH first_purchase AS (
    SELECT
        sp.student_id,
        MIN(sp.date_purchased) AS first_purchase_date
    FROM student_purchases sp
    WHERE sp.student_id IS NOT NULL
      AND sp.date_purchased IS NOT NULL
      AND (sp.date_refunded IS NULL OR sp.date_refunded < sp.date_purchased)
    GROUP BY sp.student_id
),
time_to_purchase AS (
    SELECT
        si.student_id,
        si.date_registered,
        fp.first_purchase_date,
        DATEDIFF(fp.first_purchase_date, si.date_registered) AS days_to_purchase
    FROM student_info si
    JOIN first_purchase fp
      ON si.student_id = fp.student_id
    WHERE si.date_registered IS NOT NULL
)
SELECT
    COUNT(*) AS converted_students,
    ROUND(AVG(days_to_purchase), 2) AS avg_days_to_purchase,
    MIN(days_to_purchase) AS min_days_to_purchase,
    MAX(days_to_purchase) AS max_days_to_purchase
FROM time_to_purchase
LIMIT 1;

-- Optional distribution by weekly buckets
WITH first_purchase AS (
    SELECT
        sp.student_id,
        MIN(sp.date_purchased) AS first_purchase_date
    FROM student_purchases sp
    WHERE sp.student_id IS NOT NULL
      AND sp.date_purchased IS NOT NULL
      AND (sp.date_refunded IS NULL OR sp.date_refunded < sp.date_purchased)
    GROUP BY sp.student_id
),
time_to_purchase AS (
    SELECT
        DATEDIFF(fp.first_purchase_date, si.date_registered) AS days_to_purchase
    FROM student_info si
    JOIN first_purchase fp
      ON si.student_id = fp.student_id
    WHERE si.date_registered IS NOT NULL
)
SELECT
    FLOOR(days_to_purchase / 7) AS week_bucket,
    COUNT(*) AS students
FROM time_to_purchase
GROUP BY FLOOR(days_to_purchase / 7)
ORDER BY week_bucket;


/* ==========================================================
   4) CERTIFICATE PERFORMANCE
   Links certification with engagement and purchasing outcomes.
   ========================================================== */
WITH watch_time AS (
    SELECT
        svw.student_id,
        SUM(COALESCE(svw.seconds_watched, 0)) AS total_seconds_watched
    FROM student_video_watched svw
    WHERE svw.student_id IS NOT NULL
    GROUP BY svw.student_id
),
purchase_status AS (
    SELECT
        sp.student_id,
        COUNT(*) AS purchases,
        SUM(CASE WHEN sp.date_refunded IS NOT NULL AND sp.date_refunded >= sp.date_purchased THEN 1 ELSE 0 END) AS refunded_purchases
    FROM student_purchases sp
    WHERE sp.student_id IS NOT NULL
    GROUP BY sp.student_id
),
certificate_counts AS (
    SELECT
        sc.student_id,
        COUNT(*) AS certificates_earned
    FROM student_certificates sc
    WHERE sc.student_id IS NOT NULL
    GROUP BY sc.student_id
),
student_rollup AS (
    SELECT
        si.student_id,
        COALESCE(cc.certificates_earned, 0) AS certificates_earned,
        COALESCE(wt.total_seconds_watched, 0) AS total_seconds_watched,
        COALESCE(ps.purchases, 0) AS purchases,
        COALESCE(ps.refunded_purchases, 0) AS refunded_purchases
    FROM student_info si
    LEFT JOIN certificate_counts cc ON si.student_id = cc.student_id
    LEFT JOIN watch_time wt ON si.student_id = wt.student_id
    LEFT JOIN purchase_status ps ON si.student_id = ps.student_id
)
SELECT
    CASE WHEN certificates_earned > 0 THEN 'certified' ELSE 'not_certified' END AS certificate_group,
    COUNT(*) AS students,
    ROUND(AVG(total_seconds_watched), 2) AS avg_seconds_watched,
    ROUND(AVG(purchases), 2) AS avg_purchases,
    ROUND(AVG(refunded_purchases), 2) AS avg_refunded_purchases,
    ROUND(SUM(CASE WHEN purchases > 0 THEN 1 ELSE 0 END) / COUNT(*), 4) AS purchase_rate
FROM student_rollup
GROUP BY certificate_group
ORDER BY certificate_group;
