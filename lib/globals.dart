library my_prj.globals;

// Number of potatoes detected
int totalPDT = 0;

// Information relative to the height
int meanHeight = 0;
int medHeight = 0;
List<num> allHeights = [];

// Information relative to te diameter
int meanDiameter = 0;
int medDiameter = 0;
List<num> allDiameters = [];

// Objects of height greater than 4 inches (number and %)
int gt4po = 0;
int gt17po = 0;

// Ojects of diameter greater than 1.7 inches (number and %)
String gt4pc = '';
String gt17pc = '';

// Information relative to the weight
int meanWeight = 0;
int medWeight = 0;
int totalWeight = 0;
List<num> allWeights = [];

// Information relative to the calibres
Map calibresMap = {
  'a) 3 po+': 0,
  'b) 3 po': 0,
  'c) 2 1/2 po': 0,
  'd) 2 1/4 po': 0,
  'e) 2 po': 0,
  'f) 1 7/8 po': 0,
  'g) 1 3/4 po': 0
};
