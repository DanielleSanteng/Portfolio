SELECT DISTINCT PlayerOriginID, PlayerDoB, CAST(DATEDIFF(DD, PlayerDoB, GETDATE())/365.25 AS INT) AS Age, EventName
FROM PlayerDim INNER JOIN PlayerInGameDim ON PlayerDim.PlayerID = PlayerInGameDim.PlayerID INNER JOIN GameFact ON GameFact.GameID = PlayerInGameDim.GameID INNER JOIN EventFact ON EventFact.EventID = GameFact.EventID
INNER JOIN DateDim ON DateDim.DateID = EventFact.DateID INNER JOIN EventDim ON EventFact.EventID = EventDim.EventID 
WHERE YEAR(DateValue) = 2021
ORDER BY [Age] ASC;

SELECT CONVERT(varchar, DateValue, 3), PlayerID, ChampionItemName, ChampionItemPrice 
FROM Datedim INNER JOIN ;

SELECT YEAR(DateValue) AS [Year], DATENAME(MONTH, DateValue) AS [Month], COUNT(PlayerInGameDim.PlayerID) AS [No. of Players]
FROM GameFact INNER JOIN DateDim ON GameFact.DateID = DateDim.DateID INNER JOIN GameDim ON
GameDim.GameID = GameFact.GameID INNER JOIN PlayerInGameDim ON PlayerInGameDim.GameID = GameDim.GameID INNER JOIN PlayerDim ON PlayerDim.PlayerID = PlayerInGameDim.PlayerID
GROUP BY ROLLUP (YEAR(DateValue), DATENAME(MONTH, DateValue)) ORDER BY YEAR(DateValue), DATENAME(MONTH, DateValue) DESC;

SELECT Country, FORMAT(SUM(TicketsSold), 'N0') AS [Total Tickets Sold]
, FORMAT((SUM(TicketsSoldPND)), 'N') AS [Total Sales Amount PND], FORMAT(SUM(TicketsRefunded), 'N0') AS [Total Tickets Refunded],
FORMAT(SUM(TicketsRefundedPND), 'N') AS [Total Refund Amount PND],
CAST((CAST(SUM(TicketsRefunded) AS FLOAT)/SUM(TicketsSold)) AS decimal(10,2)) * 100 AS [Per
centage Refunded],
FORMAT(SUM(TicketsSoldPND) - SUM(TicketsRefundedPND), 'N') AS [Total Profit]
FROM EventFact INNER JOIN RefundFact ON EventFact.TicketID = RefundFact.TicketID
INNER JOIN GameFact ON GameFact.EventID = EventFact.EventID INNER JOIN StadiumDim ON StadiumDim.StadiumID = GameFact.StadiumID
INNER JOIN LocationDim ON LocationDim.LocationID = StadiumDim.StadiumLocationID GROUP BY Country;

SELECT [Month], figures, clothing, pins, statues, accessories, [board games], plush, [art a
nd book]
FROM
(SELECT MerchandiseType, DATENAME(MONTH, DateValue) [Month], MONTH(DateValue) AS [Month Order], SUM(EventFact.MerchandiseSoldPND) +
SUM(OnlineSalesFact.MerchandiseSoldPND) AS [Total Merchandise Sales PND]
FROM EventFact INNER JOIN OnlineSalesFact ON EventFact.MerchandiseID = OnlineSalesFact.MerchandiseID INNER JOIN DateDim ON EventFact.DateID = DateDim.DateID INNER JOIN MerchandiseDim
ON OnlineSalesFact.MerchandiseID = MerchandiseDim.MerchandiseID
WHERE YEAR(DateValue) = 2021
GROUP BY MerchandiseType, DATENAME(MONTH, DateValue), MONTH(DateValue))
UP PIVOT(SUM([Total Merchandise Sales PND])
FOR MerchandiseType IN ([figures], [clothing], [pins], [statues], [accessories], [board gam
es], [plush], [art and book]))
AS MerchandisePvt
ORDER BY [Month Order];

SELECT DISTINCT PlayerOriginID, PlayerDoB, CAST(DATEDIFF(DD, PlayerDoB, GETDATE())/365.25 AS INT) AS Age, EventName
FROM PlayerDim INNER JOIN PlayerInGameDim ON PlayerDim.PlayerID = PlayerInGameDim.PlayerID
INNER JOIN GameFact ON GameFact.GameID = PlayerInGameDim.GameID INNER JOIN EventFact ON EventFact.EventID = GameFact.EventID
INNER JOIN DateDim ON DateDim.DateID = EventFact.DateID INNER JOIN EventDim ON EventFact.EventID = EventDim.EventID
WHERE YEAR(DateValue) = 2021
ORDER BY [Age] ASC;

SELECT TOP 20 PlayerGameName, CAST(FORMAT(ROUND(CAST((PRKills + PRAssists) AS DECIMAL (4,2))/(PRDeaths + 1), 2), 'N2') AS FLOAT) AS [Kill Assist to Death Ratio]
FROM PlayerDim INNER JOIN PlayerInGameDim ON PlayerInGameDim.PlayerID = PlayerDim.PlayerID
INNER JOIN PersonalRecordDim ON PlayerInGameDim.PRID = PersonalRecordDim.PRID
ORDER BY [Kill Assist to Death Ratio] DESC;
