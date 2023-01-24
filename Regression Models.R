> max(Assignment_1_data_set$Claims, na.rm = TRUE) 
[1] 3338
> ZoneClaims <- data.frame(Assignment_1_data_set$Zone, Assignment_1_data_set$Claims)
> ZoneClaims %>% group_by(Assignment_1_data_set.Zone) %>% summarise_each(funs(sum))
Warning: `summarise_each_()` was deprecated in dplyr 0.7.0.
Please use `across()` instead.
This warning is displayed once every 8 hours.
Call `lifecycle::last_warnings()` to see where this warning was generated.
Warning: `funs()` was deprecated in dplyr 0.8.0.
Please use a list of either functions or lambdas: 
  
  # Simple named list: 
  list(mean = mean, median = median)

# Auto named with `tibble::lst()`: 
tibble::lst(mean, median)

# Using lambdas
list(~ mean(., trim = .2), ~ median(., na.rm = TRUE))
This warning is displayed once every 8 hours.
Call `lifecycle::last_warnings()` to see where this warning was generated.
# A tibble: 7 x 2
Assignment_1_data_set.Zone Assignment_1_data_set.Claims
<dbl>                        <dbl>
  1                          1                        23174
2                          2                        21302
3                          3                        19938
4                          4                        31913
5                          5                         5962
6                          6                        10262
7                          7                          620
> library(readxl)
> Assignment1ZoneClaims <- read_excel("MSc Data Analytics/Assignments/Assignment1ZoneClaims.xlsx")
> View(Assignment1ZoneClaims)
> ZoneClaimsBar <- ggplot (Assignment1ZoneClaims, aes(Zone, Claims))
> ZoneClaimsBar + stat_summary(geom = "bar", fill = "pink", colour = "black") + ggtitle("Geographical Zone vs Total no. of Claims") + labs(x = "Geographical Zone", y = "Total no. of Claims") + scale_x_continuous(breaks = breaks_width(1))
No summary function supplied, defaulting to `mean_se()`
> MakeClaims <- data.frame(Assignment_1_data_set$Make, Assignment_1_data_set$Claims)
> MakeClaims %>% group_by(Assignment_1_data_set.Make) %>% summarise_each(funs(sum))
# A tibble: 9 x 2
Assignment_1_data_set.Make Assignment_1_data_set.Claims
<dbl>                        <dbl>
  1                          1                        11622
2                          2                         2747
3                          3                         1847
4                          4                         2065
5                          5                         3094
6                          6                         4664
7                          7                         2180
8                          8                         1103
9                          9                        83849
> library(readxl)
> Assignment1MakeClaims <- read_excel("MSc Data Analytics/Assignments/Assignment1MakeClaims.xlsx")
> View(Assignment1MakeClaims)
> MakeClaimsBar <- ggplot (Assignment1MakeClaims, aes(Make, Claims))
> MakeClaimsBar + stat_summary(geom = "bar", fill = "lightblue", colour = "black") + ggtitle("Make of Car vs Total no. of Claims") + labs(x = "Make of Car", y = "Total no. of Claims") + scale_x_continuous(breaks = breaks_width(1))
No summary function supplied, defaulting to `mean_se()`
> MakeHistogram <- ggplot(Assignment_1_data_set, aes(Make))
> MakeHistogram + geom_histogram(fill = "yellow", colour = "black", binwidth = 0.5) + labs(x = "Make of Car") + ggtitle ("Frequency of Make of Car") + scale_x_continuous(breaks = breaks_width(1))
> MakePayment <- ggplot(Assignment_1_data_set, aes(Make, Payment))
> MakePayment + stat_summary(geom = "bar", fill = "purple", colour = "black") + ggtitle("Make of Car vs Total Payment") + labs(x = "Make of Car", y = "Total Payment") + scale_x_continuous(breaks = breaks_width(1))
No summary function supplied, defaulting to `mean_se()`
> KilometresClaims <- data.frame(Assignment_1_data_set$Kilometres, Assignment_1_data_set$Claims)
> KilometresClaims %>% group_by(Assignment_1_data_set.Kilometres) %>% summarise_each(funs(sum))
# A tibble: 5 x 2
Assignment_1_data_set.Kilometres Assignment_1_data_set.Claims
<dbl>                        <dbl>
  1                                1                        33186
2                                2                        39371
3                                3                        23885
4                                4                         9025
5                                5                         7704
> library(readxl)
> Assignment1ClaimsKilometres <- read_excel("MSc Data Analytics/Assignments/Assignment1ClaimsKilometres.xlsx")
> View(Assignment1ClaimsKilometres)
> KilometresClaimsBar <- ggplot (Assignment1ClaimsKilometres, aes(Kilometres, Claims))
> KilometresClaimsBar + stat_summary(geom = "bar", fill = "orange", colour = "black") + ggtitle("Kilometres Travelled per year vs Total no. of Claims") + labs(x = "Kilometres Travelled per Year", y = "Total no. of Claims") + scale_x_continuous(breaks = breaks_width(1))
No summary function supplied, defaulting to `mean_se()`
> ZoneMake <- data.frame(Assignment_1_data_set$Zone, Assignment_1_data_set$Make)
> ZoneMake %>% group_by(Assignment_1_data_set.Zone) %>% summarise_each(funs(sum))
# A tibble: 7 x 2
Assignment_1_data_set.Zone Assignment_1_data_set.Make
<dbl>                      <dbl>
  1                          1                       1575
2                          2                       1575
3                          3                       1575
4                          4                       1575
5                          5                       1567
6                          6                       1575
7                          7                       1450
> library(readxl)
> Assignment1ZoneMake <- read_excel("MSc Data Analytics/Assignments/Assignment1ZoneMake.xlsx")
> View(Assignment1ZoneMake)
> ZoneMake <- ggplot(Assignment1ZoneMake, aes(Zone, Make))
> ZoneMake + stat_summary(geom = "bar", fill = "lightblue", colour = "black") + ggtitle("Geographical Zone vs Make of Car") + labs(x = "Geographical Zone", y = "Make of Car") + scale_x_continuous(breaks = breaks_width(1))
No summary function supplied, defaulting to `mean_se()`
> max(Assignment_1_data_set$Claims)
[1] 3338
> min(Assignment_1_data_set$Claims)
[1] 0
> median(Assignment_1_data_set$Claims)
[1] 5
> IQR(Assignment_1_data_set$Claims)
[1] 20
> mean(Assignment_1_data_set$Claims)
[1] 51.86572
> ClaimsBoxplot <- ggplot(Assignment_1_data_set, aes(Claims))
> ClaimsBoxplot + geom_boxplot() +labs(x = "Number of Claims") + ggtitle("Number of Claims Dispersion") + theme(axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + scale_x_continuous(trans='log2', breaks = trans_breaks("log2", function(x) 2^x), labels = trans_format("log2", math_format(2^.x)))
Warning: Transformation introduced infinite values in continuous x-axis
Warning: Removed 385 rows containing non-finite values (stat_boxplot).
> BonusClaims <- data.frame(Assignment_1_data_set$Bonus, Assignment_1_data_set$Claims)
> BonusClaims %>% group_by(Assignment_1_data_set.Bonus) %>% summarise_each(funs(sum))
# A tibble: 7 x 2
Assignment_1_data_set.Bonus Assignment_1_data_set.Claims
<dbl>                        <dbl>
  1                           1                        19189
2                           2                        10681
3                           3                         7742
4                           4                         6309
5                           5                         7143
6                           6                        12582
7                           7                        49525
> library(readxl)
> Assignment1BonusClaims <- read_excel("MSc Data Analytics/Assignments/Assignment1BonusClaims.xlsx")
> View(Assignment1BonusClaims)
> BonusClaims <- ggplot(Assignment1BonusClaims, aes(Bonus, Claims))
> BonusClaims + stat_summary(geom = "bar", fill = "lightgreen", colour = "black") + ggtitle("No Claims Bonus and Total no. of Claims") + labs(x = "No Claims Bonus", y = "Total no. of Claims") + scale_x_continuous(breaks = breaks_width(1))
No summary function supplied, defaulting to `mean_se()`
> payment <- Assignment_1_data_set$Payment / 1000
> Assignment_1_data_set$payment <- Assignment_1_data_set$Payment / 1000
> ClaimsPayment <- ggplot(Assignment_1_data_set, aes(Claims, payment))
> ClaimsPayment + geom_point()+ labs(x="No. of Claims", y= "Total Payment (1000s)") + ggtitle("No. of Claims vs Total Payment") + scale_x_continuous(breaks = breaks_width(500)) + scale_y_continuous(breaks = breaks_width(2500)) + geom_smooth(method = "lm")
`geom_smooth()` using formula 'y ~ x'
> mean(Assignment_1_data_set$Kilometres)
[1] 2.985793
> median(Assignment_1_data_set$Kilometres)
[1] 3
> Mode <- function(v) { uniqv <- unique(v) uniqv[which.max(tabulate(match(v, uniqv)))]}
Error: unexpected symbol in "Mode <- function(v) { uniqv <- unique(v) uniqv"
> Mode <- function(v) { uniqv <- unique(v) 
+ uniqv[which.max(tabulate(match(v, uniqv)))]}
> Kilo <- c(Assignment_1_data_set$Kilometres)
> result <- Mode(Kilo)
> print(result)
[1] 2
> mean(Assignment_1_data_set$Zone)
[1] 3.970211
> median(Assignment_1_data_set$Zone)
[1] 4
> mean(Assignment_1_data_set$Bonus)
[1] 4.015124
> median(Assignment_1_data_set$Bonus)
[1] 4
> B <- c(Assignment_1_data_set$Bonus)
> result <- Mode(B)
> print(result)
[1] 6
> mean(Assignment_1_data_set$Make)
[1] 4.991751
> median(Assignment_1_data_set$Make)
[1] 5
> M <- c(Assignment_1_data_set$Make)
> result <- Mode(M)
> print(result)
[1] 1
> mean(Assignment_1_data_set$Insured)
[1] 1092.195
> quantile(Assignment_1_data_set$Insured) 
0%         25%         50%         75%        100% 
0.0100     21.6100     81.5250    389.7825 127687.2700 
> IQR(Assignment_1_data_set$Insured)
[1] 368.1725
> mean(Assignment_1_data_set$Claims)
[1] 51.86572
> quantile(Assignment_1_data_set$Claims)
0%  25%  50%  75% 100% 
0    1    5   21 3338 
> IQR(Assignment_1_data_set$Claims)
[1] 20
> mean(Assignment_1_data_set$Payment)
[1] 257007.6
> quantile(Assignment_1_data_set$Payment)
0%         25%         50%         75%        100% 
0.00     2988.75    27403.50   111953.75 18245026.00 
> IQR(Assignment_1_data_set$Payment)
[1] 108965
> AssignmentMatrix <- as.matrix(Assignment_1_data_set)
> Hmisc::rcorr(AssignmentMatrix, type = "pearson")
Kilometres  Zone Bonus  Make Insured Claims Payment payment
Kilometres       1.00 -0.01  0.01  0.00   -0.11  -0.13   -0.12   -0.12
Zone            -0.01  1.00  0.01 -0.01   -0.06  -0.11   -0.10   -0.10
Bonus            0.01  0.01  1.00  0.00    0.17   0.11    0.12    0.12
Make             0.00 -0.01  0.00  1.00    0.19   0.25    0.24    0.24
Insured         -0.11 -0.06  0.17  0.19    1.00   0.91    0.93    0.93
Claims          -0.13 -0.11  0.11  0.25    0.91   1.00    1.00    1.00
Payment         -0.12 -0.10  0.12  0.24    0.93   1.00    1.00    1.00
payment         -0.12 -0.10  0.12  0.24    0.93   1.00    1.00    1.00

n= 2182 


P
Kilometres Zone   Bonus  Make   Insured Claims Payment payment
Kilometres            0.5120 0.7358 0.9008 0.0000  0.0000 0.0000  0.0000 
Zone       0.5120            0.5832 0.8076 0.0065  0.0000 0.0000  0.0000 
Bonus      0.7358     0.5832        0.9200 0.0000  0.0000 0.0000  0.0000 
Make       0.9008     0.8076 0.9200        0.0000  0.0000 0.0000  0.0000 
Insured    0.0000     0.0065 0.0000 0.0000         0.0000 0.0000  0.0000 
Claims     0.0000     0.0000 0.0000 0.0000 0.0000         0.0000  0.0000 
Payment    0.0000     0.0000 0.0000 0.0000 0.0000  0.0000         0.0000 
payment    0.0000     0.0000 0.0000 0.0000 0.0000  0.0000 0.0000         
> PaymentClaimsModel <- lm(Payment ~ Claims, data = Assignment_1_data_set)
> summary(PaymentClaimsModel)

Call:
  lm(formula = Payment ~ Claims, data = Assignment_1_data_set)

Residuals:
  Min       1Q   Median       3Q      Max 
-1744858    -8545     2773    13386  1491369 

Coefficients:
  Estimate Std. Error t value Pr(>|t|)    
(Intercept) -3362.29    2154.79   -1.56    0.119    
Claims       5020.08      10.35  485.11   <2e-16 ***
  ---
  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 97480 on 2180 degrees of freedom
Multiple R-squared:  0.9908,	Adjusted R-squared:  0.9908 
F-statistic: 2.353e+05 on 1 and 2180 DF,  p-value: < 2.2e-16

> PaymentInsuredModel <- lm(Payment ~ Insured, data = Assignment_1_data_set)
> summary(PaymentInsuredModel)

Call:
  lm(formula = Payment ~ Insured, data = Assignment_1_data_set)

Residuals:
  Min       1Q   Median       3Q      Max 
-5946157   -75828   -70260   -30246  5343552 

Coefficients:
  Estimate Std. Error t value Pr(>|t|)    
(Intercept) 73852.388   7971.250   9.265   <2e-16 ***
  Insured       167.695      1.383 121.266   <2e-16 ***
  ---
  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 365600 on 2180 degrees of freedom
Multiple R-squared:  0.8709,	Adjusted R-squared:  0.8708 
F-statistic: 1.471e+04 on 1 and 2180 DF,  p-value: < 2.2e-16

> PaymentInsuredModel <- lm(Payment ~ Insured + Make + Bonus + Zone, data = Assignment_1_data_set)
> summary(PaymentInsuredModel)

Call:
  lm(formula = Payment ~ Insured + Make + Bonus + Zone, data = Assignment_1_data_set)

Residuals:
  Min       1Q   Median       3Q      Max 
-5767596  -112618   -42403    37469  5115109 

Coefficients:
  Estimate Std. Error t value Pr(>|t|)    
(Intercept) 102111.357  27094.618   3.769 0.000168 ***
  Insured        165.803      1.383 119.925  < 2e-16 ***
  Make         28341.234   2978.331   9.516  < 2e-16 ***
  Bonus       -17386.885   3838.158  -4.530 6.22e-06 ***
  Zone        -24647.137   3812.343  -6.465 1.25e-10 ***
  ---
  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 353400 on 2177 degrees of freedom
Multiple R-squared:  0.8795,	Adjusted R-squared:  0.8793 
F-statistic:  3974 on 4 and 2177 DF,  p-value: < 2.2e-16

> PaymentModel <- lm(Payment ~ Claims + Make + Bonus + Zone, data = Assignment_1_data_set)
> summary(PaymentModel)

Call:
  lm(formula = Payment ~ Claims + Make + Bonus + Zone, data = Assignment_1_data_set)

Residuals:
  Min       1Q   Median       3Q      Max 
-1711722   -19583      994    22073  1448528 

Coefficients:
  Estimate Std. Error t value Pr(>|t|)    
(Intercept) -36263.85    7291.05  -4.974 7.08e-07 ***
  Claims        5031.37      10.62 473.981  < 2e-16 ***
  Make         -3555.90     817.51  -4.350 1.43e-05 ***
  Bonus         6642.46    1028.37   6.459 1.29e-10 ***
  Zone          5892.88    1035.49   5.691 1.43e-08 ***
  ---
  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 95490 on 2177 degrees of freedom
Multiple R-squared:  0.9912,	Adjusted R-squared:  0.9912 
F-statistic: 6.134e+04 on 4 and 2177 DF,  p-value: < 2.2e-16

> anova(PaymentClaimsModel, PaymentModel)
Analysis of Variance Table

Model 1: Payment ~ Claims
Model 2: Payment ~ Claims + Make + Bonus + Zone
Res.Df        RSS Df  Sum of Sq      F    Pr(>F)    
1   2180 2.0716e+13                                   
2   2177 1.9849e+13  3 8.6725e+11 31.707 < 2.2e-16 ***
  ---
  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
> standardresidual <- rstandard(PaymentModel)
> PaymentResiduals <- data.frame(standardresidual)
> PaymentResiduals$large.residual <- PaymentResiduals$standardresidual > 2 | PaymentResiduals$standardresidual < -2
> Assignment_1_data_set <- Assignment_1_data_set[ ,-8]
> sum(PaymentResiduals$large.residual)
[1] 66
> 66/2182 
[1] 0.03024748
> Assignment_1_data_set$cook <- cooks.distance(PaymentModel)
> Assignment_1_data_set[Assignment_1_data_set$cook > 1, ]
# A tibble: 3 x 8
Kilometres  Zone Bonus  Make Insured Claims  Payment  cook
<dbl> <dbl> <dbl> <dbl>   <dbl>  <dbl>    <dbl> <dbl>
  1          1     1     1     9   9998.   1704  6805992  2.35
2          1     4     7     9 127687.   2894 15540162  2.43
3          2     4     7     9 121293.   3338 18245026  7.69
> plot(PaymentModel, 4,id.n = 5)
> Assignment_1_data_set <- Assignment_1_data_set[-c(9, 252, 691), ]
> PaymentModel <- lm(Payment ~ Claims + Make + Bonus + Zone, data = Assignment_1_data_set)
> Summary(PaymentModel)
Error in (function (classes, fdef, mtable)  : 
            unable to find an inherited method for function 'Summary' for signature '"lm"'
          > summary(PaymentModel)
          
          Call:
            lm(formula = Payment ~ Claims + Make + Bonus + Zone, data = Assignment_1_data_set)
          
          Residuals:
            Min      1Q  Median      3Q     Max 
          -765085  -19029   -1216   17497  949691 
          
          Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
          (Intercept) -30044.2     6020.1  -4.991 6.49e-07 ***
            Claims        4955.6       10.1 490.414  < 2e-16 ***
            Make         -2418.8      677.0  -3.573 0.000361 ***
            Bonus         5920.0      849.7   6.967 4.28e-12 ***
            Zone          4468.1      856.1   5.219 1.97e-07 ***
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
          
          Residual standard error: 78790 on 2174 degrees of freedom
          Multiple R-squared:  0.9919,	Adjusted R-squared:  0.9918 
          F-statistic: 6.617e+04 on 4 and 2174 DF,  p-value: < 2.2e-16
          
          > PaymentClaims <- ggplot(PM, aes(Claims, Payment)) + geom_point() + stat_smooth(method = lm)
          Error in ggplot(PM, aes(Claims, Payment)) : object 'PM' not found
          > PM <- data.frame (Assignment_1_data_set$Payment, Assignment_1_data_set$Claims)
          > PaymentClaims <- ggplot(PM, aes(Claims, Payment)) + geom_point() + stat_smooth(method = lm)
          > PaymentClaims + geom_line(aes(y = lwr), color = "red", linetype = "dashed")+ geom_line(aes(y = upr), color = "red", linetype = "dashed") + scale_x_continuous(breaks = breaks_width(500)) + labs(x="No. of Claims", y= "Total Payment (1000s)") + ggtitle("Claims vs Total Payment Regression Model")
          Error in FUN(X[[i]], ...) : object 'Payment' not found
          > view(PM)
          > PaymentClaims <- ggplot(PM, aes(Assignment_1_data_set.Claims, Assignment_1_data_set.Payment)) + geom_point() + stat_smooth(method = lm)
          > PaymentClaims + geom_line(aes(y = lwr), color = "red", linetype = "dashed")+ geom_line(aes(y = upr), color = "red", linetype = "dashed") + scale_x_continuous(breaks = breaks_width(500)) + labs(x="No. of Claims", y= "Total Payment (1000s)") + ggtitle("Claims vs Total Payment Regression Model")
          Error in FUN(X[[i]], ...) : object 'lwr' not found
          > mean(vif(PaymentModel))
          [1] 1.052003
          > library(readxl)
          > Assignment_1_data_set <- read_excel("MSc Data Analytics/Assignment 1 - data set.xlsx")
          > View(Assignment_1_data_set)
          > ClaimsModel <- lm(Claims ~ Insured + Make + Bonus + Kilometres + Zone, data = Assignment_1_data_set)
          > summary(ClaimsModel)
          
          Call:
            lm(formula = Claims ~ Insured + Make + Bonus + Kilometres + Zone, 
               data = Assignment_1_data_set)
          
          Residuals:
            Min       1Q   Median       3Q      Max 
          -1214.57   -25.18    -9.41    10.04  1301.78 
          
          Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
          (Intercept) 37.1230027  7.1270679   5.209 2.08e-07 ***
            Insured      0.0318697  0.0003158 100.933  < 2e-16 ***
            Make         6.7725342  0.6755390  10.025  < 2e-16 ***
            Bonus       -4.2468101  0.8707236  -4.877 1.15e-06 ***
            Kilometres  -3.9648601  1.2255209  -3.235  0.00123 ** 
            Zone        -6.2924300  0.8647405  -7.277 4.75e-13 ***
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
          
          Residual standard error: 80.14 on 2176 degrees of freedom
          Multiple R-squared:  0.8425,	Adjusted R-squared:  0.8421 
          F-statistic:  2328 on 5 and 2176 DF,  p-value: < 2.2e-16
          
          > step(ClaimsModel, direction = "backward")
          Start:  AIC=19136.97
          Claims ~ Insured + Make + Bonus + Kilometres + Zone
          
          Df Sum of Sq      RSS   AIC
          <none>                    13976386 19137
          - Kilometres  1     67228 14043614 19145
          - Bonus       1    152792 14129178 19159
          - Zone        1    340096 14316481 19187
          - Make        1    645561 14621947 19234
          - Insured     1  65433535 79409920 22926
          
          Call:
            lm(formula = Claims ~ Insured + Make + Bonus + Kilometres + Zone, 
               data = Assignment_1_data_set)
          
          Coefficients:
            (Intercept)      Insured         Make        Bonus   Kilometres         Zone  
          37.12300      0.03187      6.77253     -4.24681     -3.96486     -6.29243  
          
          > ClaimsModel2 <- lm(Claims ~ Insured + Make + Bonus + Zone, data = Assignment_1_data_set)
          > summary
          function (object, ...) 
            UseMethod("summary")
          <bytecode: 0x000001bf599750e8>
            <environment: namespace:base>
            > summary(ClaimsModel2)
          
          Call:
            lm(formula = Claims ~ Insured + Make + Bonus + Zone, data = Assignment_1_data_set)
          
          Residuals:
            Min       1Q   Median       3Q      Max 
          -1221.61   -24.60    -9.71     9.42  1308.69 
          
          Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
          (Intercept) 25.4393382  6.1577008   4.131 3.74e-05 ***
            Insured      0.0319907  0.0003142 101.813  < 2e-16 ***
            Make         6.7295142  0.6768750   9.942  < 2e-16 ***
            Bonus       -4.3242455  0.8722850  -4.957 7.70e-07 ***
            Zone        -6.2322582  0.8664181  -7.193 8.67e-13 ***
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
          
          Residual standard error: 80.32 on 2177 degrees of freedom
          Multiple R-squared:  0.8417,	Adjusted R-squared:  0.8415 
          F-statistic:  2895 on 4 and 2177 DF,  p-value: < 2.2e-16
          
          > anova(ClaimsModel2, ClaimsModel)
          Analysis of Variance Table
          
          Model 1: Claims ~ Insured + Make + Bonus + Zone
          Model 2: Claims ~ Insured + Make + Bonus + Kilometres + Zone
          Res.Df      RSS Df Sum of Sq      F   Pr(>F)   
          1   2177 14043614                                
          2   2176 13976386  1     67228 10.467 0.001234 **
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
          > standardresidual <- rstandard(ClaimsModel)
          > ClaimsResiduals <- data.frame(standardresidual)
          > ClaimsResiduals$large.residual <- ClaimsResiduals$standardresidual > 2 | ClaimsResiduals$standardresidual < -2
          > sum(ClaimsResiduals$large.residual)
          [1] 48
          > 48/2182 
          [1] 0.02199817
          > plot(ClaimsModel, 4,id.n = 5)
          > Assignment_1_data_set$Claimscook <- cooks.distance(ClaimsModel)
          > Assignment_1_data_set[Assignment_1_data_set$Claimscook > 1,]
          # A tibble: 2 x 8
          Kilometres  Zone Bonus  Make Insured Claims  Payment Claimscook
          <dbl> <dbl> <dbl> <dbl>   <dbl>  <dbl>    <dbl>      <dbl>
            1          1     4     7     9 127687.   2894 15540162      15.6 
          2          2     4     7     9 121293.   3338 18245026       2.85
          > Assignment_1_data_set <- Assignment_1_data_set[-c(252, 691), ]
          > Assignment_1_data_set <- Assignment_1_data_set[ , -8]
          > ClaimsModel <- lm(Claims ~ Insured + Make + Bonus + Kilometres + Zone, data = Assignment_1_data_set)
          > summary(ClaimsModel)
          
          Call:
            lm(formula = Claims ~ Insured + Make + Bonus + Kilometres + Zone, 
               data = Assignment_1_data_set)
          
          Residuals:
            Min      1Q  Median      3Q     Max 
          -649.38  -21.16   -6.87    9.75 1253.47 
          
          Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
          (Intercept) 37.9076954  6.2861696   6.030 1.92e-09 ***
            Insured      0.0379422  0.0003748 101.242  < 2e-16 ***
            Make         5.2078463  0.5992953   8.690  < 2e-16 ***
            Bonus       -5.9936164  0.7713337  -7.770 1.20e-14 ***
            Kilometres  -2.3727848  1.0831155  -2.191   0.0286 *  
            Zone        -5.2487649  0.7639124  -6.871 8.30e-12 ***
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
          
          Residual standard error: 70.69 on 2174 degrees of freedom
          Multiple R-squared:  0.8445,	Adjusted R-squared:  0.8441 
          F-statistic:  2361 on 5 and 2174 DF,  p-value: < 2.2e-16
          
          > mean(vif(ClaimsModel))
          [1] 1.043319
          > library(readxl)
          > Assignment_1_data_set <- read_excel("MSc Data Analytics/Assignment 1 - data set.xlsx")
          > View(Assignment_1_data_set)
          > Case1 <- data.frame(Kilometres = c(2), Zone = c(5), Bonus = c(3), Make = c(3), Insured = c(4621))
          > predict(ClaimsModel, newdata = Case1)
          1 
          179.892 
          > predict(ClaimsModel, newdata = Case1, interval = "confidence")
          fit      lwr      upr
          1 179.892 174.0724 185.7116
          > Case1$Claims <- 180
          > predict(PaymentModel, newdata = Case1)
          1 
          894807.9 
          > predict(PaymentModel, newdata = Case1, interval = "confidence")
          fit    lwr      upr
          1 894807.9 888726 900889.7
          > Case2 <- data.frame(Kilometres = c(2), Zone = c(2), Bonus = c(1), Make = c(9), Insured = c(9500)) 
          > predict(ClaimsModel, newdata = Case2)
          1 
          423.9927 
          > predict(ClaimsModel, newdata = Case2, interval = "confidence")
          fit      lwr      upr
          1 423.9927 414.2988 433.6866
          > Case2$Claims <- 424
          > predict(PaymentModel, newdata = Case2)
          1 
          2064217 
          > predict(PaymentModel, newdata = Case2, interval = "confidence")
          fit     lwr     upr
          1 2064217 2053702 2074733
          > mean(17500,25416)
          [1] 17500
          > (17500 + 25416)/2
          [1] 21458
          > Case3 <- data.frame(Kilometres = c(4), Zone = c(2), Bonus = c(5), Make = c(3), Insured = c(21458))
          > predict(ClaimsModel, newdata = Case3)
          1 
          817.7387 
          > predict(ClaimsModel, newdata = Case3, interval = "confidence")
          fit      lwr      upr
          1 817.7387 801.5158 833.9616
          > Case3$Claims <- 818
          > predict(PaymentModel, newdata = Case3)
          1 
          4054916 
          > InsuredModel <- lm(Insured ~ Payment + Make + Bonus + Kilometres + Zone, data = Assignment_1_data_set)
          > summary(InsuredModel)
          
          Call:
            lm(formula = Insured ~ Payment + Make + Bonus + Kilometres + 
                 Zone, data = Assignment_1_data_set)
          
          Residuals:
            Min     1Q Median     3Q    Max 
          -26139   -229    142    463  46457 
          
          Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
          (Intercept) -8.265e+02  1.766e+02  -4.680 3.04e-06 ***
            Payment      5.239e-03  4.405e-05 118.925  < 2e-16 ***
            Make        -9.530e+01  1.697e+01  -5.614 2.23e-08 ***
            Bonus        1.527e+02  2.144e+01   7.122 1.44e-12 ***
            Kilometres   3.355e+00  3.042e+01   0.110    0.912    
          Zone         1.070e+02  2.152e+01   4.972 7.15e-07 ***
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
          
          Residual standard error: 1987 on 2176 degrees of freedom
          Multiple R-squared:  0.8771,	Adjusted R-squared:  0.8768 
          F-statistic:  3106 on 5 and 2176 DF,  p-value: < 2.2e-16
          
          > step(InsuredModel, direction = "backward")
          Start:  AIC=33147.64
          Insured ~ Payment + Make + Bonus + Kilometres + Zone
          
          Df  Sum of Sq        RSS   AIC
          - Kilometres  1 4.8026e+04 8.5903e+09 33146
          <none>                     8.5903e+09 33148
          - Zone        1 9.7583e+07 8.6879e+09 33170
          - Make        1 1.2443e+08 8.7147e+09 33177
          - Bonus       1 2.0025e+08 8.7905e+09 33196
          - Payment     1 5.5833e+10 6.4424e+10 37542
          
          Step:  AIC=33145.65
          Insured ~ Payment + Make + Bonus + Zone
          
          Df  Sum of Sq        RSS   AIC
          <none>                  8.5903e+09 33146
          - Zone     1 9.7539e+07 8.6879e+09 33168
          - Make     1 1.2440e+08 8.7147e+09 33175
          - Bonus    1 2.0050e+08 8.7908e+09 33194
          - Payment  1 5.6750e+10 6.5341e+10 37571
          
          Call:
            lm(formula = Insured ~ Payment + Make + Bonus + Zone, data = Assignment_1_data_set)
          
          Coefficients:
            (Intercept)      Payment         Make        Bonus         Zone  
          -8.165e+02    5.238e-03   -9.524e+01    1.527e+02    1.070e+02  
          
          > InsuredModel2 <- lm(Insured ~ Payment + Make + Bonus + Zone, data = Assignment_1_data_set)
          > summary(InsuredModel2)
          
          Call:
            lm(formula = Insured ~ Payment + Make + Bonus + Zone, data = Assignment_1_data_set)
          
          Residuals:
            Min     1Q Median     3Q    Max 
          -26138   -231    142    466  46459 
          
          Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
          (Intercept) -8.165e+02  1.518e+02  -5.380 8.27e-08 ***
            Payment      5.238e-03  4.368e-05 119.925  < 2e-16 ***
            Make        -9.524e+01  1.696e+01  -5.615 2.22e-08 ***
            Bonus        1.527e+02  2.143e+01   7.128 1.38e-12 ***
            Zone         1.070e+02  2.151e+01   4.972 7.15e-07 ***
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
          
          Residual standard error: 1986 on 2177 degrees of freedom
          Multiple R-squared:  0.8771,	Adjusted R-squared:  0.8769 
          F-statistic:  3884 on 4 and 2177 DF,  p-value: < 2.2e-16
          
          > step(InsuredModel2, direction = "backward")
          Start:  AIC=33145.65
          Insured ~ Payment + Make + Bonus + Zone
          
          Df  Sum of Sq        RSS   AIC
          <none>                  8.5903e+09 33146
          - Zone     1 9.7539e+07 8.6879e+09 33168
          - Make     1 1.2440e+08 8.7147e+09 33175
          - Bonus    1 2.0050e+08 8.7908e+09 33194
          - Payment  1 5.6750e+10 6.5341e+10 37571
          
          Call:
            lm(formula = Insured ~ Payment + Make + Bonus + Zone, data = Assignment_1_data_set)
          
          Coefficients:
            (Intercept)      Payment         Make        Bonus         Zone  
          -8.165e+02    5.238e-03   -9.524e+01    1.527e+02    1.070e+02  
          
          > standardresidual <- rstandard(InsuredModel2) 
          > Insuredstandardresidual <- rstandard(InsuredModel2) 
          > InsuredResiduals <- data.frame(Insuredstandardresidual) 
          > InsuredResiduals$large.residual <- InsuredResiduals$Insuredstandardresidual > 2 | InsuredResiduals$Insuredstandardresidual < -2 
          > sum(InsuredResiduals$large.residual) 
          [1] 46
          > 46/2182
          [1] 0.02108158
          > Assignment_1_data_set$cook <- cooks.distance(InsuredModel2)
          > Assignment_1_data_set[Assignment_1_data_set$cook > 1,]
          # A tibble: 2 x 8
          Kilometres  Zone Bonus  Make Insured Claims  Payment  cook
          <dbl> <dbl> <dbl> <dbl>   <dbl>  <dbl>    <dbl> <dbl>
            1          1     4     7     9 127687.   2894 15540162 14.7 
          2          2     4     7     9 121293.   3338 18245026  7.02
          > Assignment_1_data_set <- Assignment_1_data_set[-c(252, 691), ]
          > InsuredModel2 <- lm(Insured ~ Payment + Make + Bonus + Zone, data = Assignment_1_data_set)
          > summary(InsuredModel2)
          
          Call:
            lm(formula = Insured ~ Payment + Make + Bonus + Zone, data = Assignment_1_data_set)
          
          Residuals:
            Min       1Q   Median       3Q      Max 
          -21698.1   -260.5     61.4    340.9  21031.8 
          
          Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
          (Intercept) -7.828e+02  1.147e+02  -6.822 1.16e-11 ***
            Payment      4.500e-03  3.820e-05 117.781  < 2e-16 ***
            Make        -5.120e+01  1.288e+01  -3.977 7.21e-05 ***
            Bonus        1.644e+02  1.620e+01  10.146  < 2e-16 ***
            Zone         6.798e+01  1.629e+01   4.172 3.14e-05 ***
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
          
          Residual standard error: 1502 on 2175 degrees of freedom
          Multiple R-squared:  0.8755,	Adjusted R-squared:  0.8753 
          F-statistic:  3824 on 4 and 2175 DF,  p-value: < 2.2e-16
          
          > mean(vif(InsuredModel2))
          [1] 1.050029
