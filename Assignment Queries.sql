

SELECT DISTINCT PlayerOriginID, PlayerDoB, CAST(DATEDIFF(DD, PlayerDoB, GETDATE())/365.25 AS INT) AS Age, EventName
FROM PlayerDim INNER JOIN PlayerInGameDim ON PlayerDim.PlayerID = PlayerInGameDim.PlayerID INNER JOIN GameFact ON GameFact.GameID = PlayerInGameDim.GameID INNER JOIN EventFact ON EventFact.EventID = GameFact.EventID
INNER JOIN DateDim ON DateDim.DateID = EventFact.DateID INNER JOIN EventDim ON EventFact.EventID = EventDim.EventID 
WHERE YEAR(DateValue) = 2021
ORDER BY [Age] ASC;

SELECT CONVERT(varchar, DateValue, 3), PlayerID, ChampionItemName, ChampionItemPrice 
FROM Datedim INNER JOIN ;
