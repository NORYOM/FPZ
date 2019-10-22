import sqlite3
import cv2
import numpy as np
from classify import Classify

conn = sqlite3.connect('./bus.db')
c = conn.cursor()

def transformFace(blobData):
	image = np.asarray(bytearray(blobData), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	scaledFace = cv2.resize(gray,(32,32))
	return scaledFace

sql_aizawa = "select id,value from face where id in (79,82,83,84,87,88,89,90,91,92,119,120,129,130,17,18,19,20,21,22,23,100,101,102,103,104,105,106,108,109,110,111,113,279,280,281,282,283,285,288,250,251,253,254,255,256,257,258,261,93,187,189,190,191,192,194,195,196,1,4,30,32,34,35,37,38,39,341,345,347,348,349,353,354,356,358,145,146,147,148,149,150,151,153,114,115,116,117,118,231,233,235,156,158,162,165,179,271,272,273,274,275,276,277,278,223,224,225,226,227,228,229,230,40,41,43,44,45,46,47,48,49,181,182,183,184,185,186,50,51,54,24,8,9,10,11,12,13,14,15,16,197,199,200,201,202,203,204,205,236,237,238,239,240,241,242,243,245,71,72,73,74,75,76,77,78,262,263,264,265,266,267,268,270,207,208,209,211,212,213,214,215,221,222,131,132,133,134,136,137,139,140,141,142,143,56,57,58,59,60,61,62,63,64,65,66,67,68,69,325,326,327,328,332,334,336,337,338,339,340,246,247,248)"
sql_asuka = "select id,value from face where id in (793,794,795,796,797,798,799,800,801,802,803,804,720,835,557,558,559,560,561,562,563,564,565,623,624,626,627,459,460,461,462,463,580,581,688,689,690,692,693,694,696,698,700,701,707,546,547,548,549,550,551,552,553,554,556,600,601,602,603,604,605,606,607,608,609,611,612,376,377,378,379,380,381,382,383,384,385,443,444,445,446,593,594,595,596,597,598,599,419,420,421,422,424,425,426,513,514,515,516,517,518,519,520,521,487,488,489,490,491,492,493,427,428,429,430,431,432,668,669,670,671,672,673,674,675,676,678,679,582,583,585,586,589,590,591,592,522,523,524,525,526,527,528,759,761,763,765,768,770,771,629,630,631,632,634,635,636,637,638,572,573,574,575,576,577,578,579,411,412,414,415,417,418,772,775,776,777,778,780,782,783,784,785,837,838,839,840,841,842,844,846,639,641,643,645,647,651,652,655,657,658,386,387,388,389,390,392,401,402,403,404,405,406,408,409,410,711,712,713,714,715,716,786,787,788,789,790,792,464,465,466,467,468,469,470,494,495,496,497,498,500,502,503,815,819,821,822,823,824,825,826,828,829,722,723,725,726,727,728,729,739,742,743,744,745,747,805,806,807,808,809,810,811,812,813,814,504,506,507,508,510,512,680,681,682,683,684,686,687,448,449,450,451,452,453,454,455,456,457,458,730,731,732,733,734,735,736,659,660,661,662,663,664,665,666,667,393,394,396,397,398,399,400,613,614,615,616,617,618,619,620,621,359,361,362,363,364,365,366,367,368,369,371,372,375,472,473,474,475,476,481,483,484,485,486,566,567,568,569,570,571,848,849,850,851,852,853,854,855,856,857,433,435,436,438,439,440,442,748,749,750,751,752,753,755,757,758)"
sql_hasimoto = "select id,value from face where id in (1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1027,156,158,162,165,179,1295,1296,1297,1298,1299,977,978,979,981,982,983,1251,1252,1253,1254,1255,1256,1257,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1138,1139,1140,1141,1144,1145,1146,1076,920,921,922,923,924,925,926,927,928,929,930,931,932,934,935,1206,1207,1208,1209,1210,1211,1212,1214,1215,1216,1217,1218,1219,1221,1258,1259,1260,1264,1265,1266,1267,1268,1269,1347,1348,1349,1350,1351,1352,1353,1355,1046,1047,1048,1049,1050,1051,1052,872,873,874,875,876,877,879,880,881,883,884,1006,1007,1008,1010,1011,1199,1200,1204,1205,1032,1033,1034,1035,1036,1037,1038,1040,1310,1311,1312,1313,1316,1041,1042,1043,1044,1045,1317,1318,1319,1320,1321,1322,1323,1324,908,909,910,911,912,913,914,915,916,917,918,919,936,937,938,939,940,941,942,943,944,945,946,947,951,1112,1113,1114,1115,1116,1117,1118,1120,1121,1122,1062,1063,1064,1065,1066,1067,1068,1069,1230,1231,1232,1233,1234,1235,1236,1270,1271,1272,1273,1274,1275,1276,1277,1278,1280,1281,1094,1095,1096,1097,1098,1099,1100,1101,984,985,986,987,988,989,990,991,992,993,994,1123,1126,1077,1078,1080,1081,1082,1083,897,898,900,901,904,905,906,907,1028,1029,1030,1031,1186,1191,1148,1149,1150,1151,1153,1154,1194,1195,1196,1197,952,953,955,956,958,1282,1283,1284,1285,1286,1287,1288,1289,1290,1292,1293,1294,1170,1171,1172,1174,1175,1176,1177,858,859,860,861,863,867,868,1155,1156,1157,1161,1162,1163,1165,1166,1168,1169,885,886,887,888,889,890,891,893,894,895,1300,1301,1302,1303,1304,1306,1307,1308,1309,1127,1129,1130,1132,1133,1136,1137,1222,1223,1224,1225,1226,1227,1228,1229,1336,1337,1338,1339,1340,1342,1344,1346,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1053,1054,1055,1056,1057,1058,1059,1060,1061,1102,1103,1105,1108,1109,1110,1111,1178,1180,1183,1185,1326,1327,1332,1335,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976)"

trainningData = []
typeSet = []
clf = Classify()
cnt = 0;

print("Fetch Data")

chk_aizawa = []
chk_asuka = []
chk_hasimoto = []

cnt_aizawa = 0
cnt_asuka = 0
cnt_hasimoto  = 0

print("Create Trainning Set")
print("--------------------")
resLst_aizawa = c.execute(sql_aizawa)
for res in resLst_aizawa:
	scaledFace = transformFace(res[1])
	trainningData.append(scaledFace)
	chk_aizawa.append(scaledFace)
	typeSet.append(0)
	cnt_aizawa += 1
print("Aizawa",cnt_aizawa)
print("--------------------")
resLst_asuka = c.execute(sql_asuka)
for res in resLst_asuka:
	scaledFace = transformFace(res[1])
	trainningData.append(scaledFace)
	chk_asuka.append(scaledFace)
	typeSet.append(1)
	cnt_asuka += 1
print("Asuka",cnt_asuka)
print("--------------------")
resLst_hasimoto = c.execute(sql_hasimoto)
for res in resLst_hasimoto:
	scaledFace = transformFace(res[1])
	trainningData.append(scaledFace)
	chk_hasimoto.append(scaledFace)
	typeSet.append(2)
	cnt_hasimoto += 1
print("Hasimoto",cnt_hasimoto)
print("--------------------")

# clf.setTrainData(trainningData)
# clf.setTypeData(typeSet)

# clf.train()
# clf.saveModule()

print("Check predict")

res_true_aizawa = 0
res_false_aizawa = 0
cnt_aizawa = 0
for face in chk_aizawa:
	cnt_aizawa += 1
	print("Aizawa: %.2f%%" % (cnt_aizawa*100/len(chk_aizawa)))
	faces = []
	faces.append(face)
	res,score = clf.chkType(faces)
	if score[0][0]<0.8:
		res_false_aizawa += 1
	else:
		res_true_aizawa += 1

res_true_asuka = 0
res_false_asuka = 0
cnt_asuka = 0
for face in chk_asuka:
	cnt_asuka += 1
	print("Asuka: %.2f%%" % (cnt_asuka*100/len(chk_asuka)))
	faces = []
	faces.append(face)
	res,score = clf.chkType(faces)
	if score[0][1]<0.8:
		res_false_asuka += 1
	else:
		res_true_asuka += 1

res_true_hasimoto = 0
res_false_hasimoto = 0
cnt_hasimoto  = 0
for face in chk_hasimoto:
	cnt_hasimoto += 1
	print("Hasimoto: %.2f%%" % (cnt_hasimoto*100/len(chk_hasimoto)))
	faces = []
	faces.append(face)
	res,score = clf.chkType(faces)
	if score[0][2]<0.8:
		res_false_hasimoto += 1
	else:
		res_true_hasimoto += 1

print("Aizawa: " , res_true_aizawa/(res_true_aizawa+res_false_aizawa))
print("Asuka: " , res_true_asuka/(res_true_asuka+res_false_asuka))
print("Hasimoto: " , res_true_hasimoto/(res_true_hasimoto+res_false_hasimoto))