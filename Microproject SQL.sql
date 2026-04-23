select * from timestamp limit 20

--1. What was the highest Nifty50 closing price in the dataset?
SELECT MAX(close) AS max_close FROM timestamp;
--2. What was the lowest Nifty50 closing price in the dataset?
SELECT MIN(close) AS min_close FROM timestamp;
--3. What is the average closing price per day?
SELECT dt, AVG(close) AS avg_close
FROM timestamp
GROUP BY dt
ORDER BY dt;
--4.. What is the total number of trading minutes recorded?
SELECT COUNT(*) AS total_minutes FROM timestamp;
--5. What is the daily maximum high price?
SELECT dt, MAX(high) AS daily_max_high
FROM timestamp
GROUP BY dt
ORDER BY dt;
--6. What is the daily minimum low price?
SELECT dt, MIN(low) AS daily_min_low
FROM timestamp
GROUP BY dt
ORDER BY dt;
--7. How many minutes had the closing price higher than opening price?
SELECT COUNT(*) AS bullish_minutes
FROM timestamp
WHERE close > open;
--8. How many minutes had the closing price lower than opening price?
SELECT COUNT(*) AS bearish_minutes
FROM timestamp
WHERE close < open;
--9. What is the average high price per hour?
SELECT FLOOR(time) AS hour, AVG(high) AS avg_high
FROM timestamp
GROUP BY FLOOR(time)
ORDER BY hour;
--10. What is the minute with the maximum single-minute price increase (close - open)?
SELECT timestamp, (close - open) AS price_change
FROM timestamp
ORDER BY price_change DESC
LIMIT 1;