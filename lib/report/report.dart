/// Class to implement a report of the data acquired
class Report {
  final String calibrationName;
  final String adresse;
  final String operatorName;
  final String date;
  final String comment;
  final int nPDT;
  final int meanH;
  final int meanD;
  final int meanW;
  final int medH;
  final int medD;
  final int medW;
  final int totalW;
  final int gt4;
  final int gt17;
  final String gt4p;
  final String gt17p;
  final Map calibres;

  Report({
    required this.calibrationName,
    required this.adresse,
    required this.operatorName,
    required this.date,
    required this.comment,
    required this.nPDT,
    required this.meanH,
    required this.meanD,
    required this.meanW,
    required this.medH,
    required this.medD,
    required this.medW,
    required this.totalW,
    required this.gt4,
    required this.gt17,
    required this.gt4p,
    required this.gt17p,
    required this.calibres,
  });
}
