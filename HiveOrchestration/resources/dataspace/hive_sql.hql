create database office;

use office;


create table employee
(ID INT, Name STRING, Dept STRING, Yoj INT, Salary INT)
row format delimited fields terminated by ','
tblproperties("skip.header.line.count"="1");

describe employee;

LOAD DATA LOCAL INPATH
'/usr/local/hive/employee.csv'
OVERWRITE INTO TABLE employee;

SELECT * FROM employee;

SELECT count(*) FROM employee;