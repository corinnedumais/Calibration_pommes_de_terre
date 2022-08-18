import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:maxiplant/globals.dart' as globals;
import 'package:stats/stats.dart';

class ValidationRoute extends StatefulWidget {
  final String imgStr;
  final int pdt;
  final int targets;

  const ValidationRoute(
      {Key? key,
      required this.imgStr,
      required this.pdt,
      required this.targets})
      : super(key: key);

  @override
  State<ValidationRoute> createState() => _ValidationRouteState();
}

class _ValidationRouteState extends State<ValidationRoute> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
            title: const Text('Validation'),
            backgroundColor: const Color.fromARGB(255, 25, 56, 26)),
        body: Container(
            height: 800,
            alignment: Alignment.center,
            padding: const EdgeInsets.all(20),
            child: Column(children: [
              Text("Pommes de terre détectées: ${widget.pdt}",
                  style: const TextStyle(fontSize: 20)),
              const SizedBox(height: 10),
              Text('Cibles détectées: ${widget.targets}',
                  style: const TextStyle(fontSize: 20)),
              const SizedBox(height: 20),
              Image.memory(base64Decode(widget.imgStr),
                  width: 400, height: 300, fit: BoxFit.cover),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                      onPressed: () {
                        Navigator.pop(context);
                      },
                      style: ElevatedButton.styleFrom(
                          backgroundColor:
                              const Color.fromARGB(255, 25, 56, 26)),
                      child: Padding(
                          padding: const EdgeInsets.all(10),
                          child: Text('Reprendre',
                              style: GoogleFonts.openSans(fontSize: 20)))),
                  const SizedBox(width: 20),
                  ElevatedButton(
                      onPressed: () async {
                        // Retrieve statistics
                        var urlStats =
                            Uri.parse('https://10.0.2.2:5000/statistics');
                        final response = await http.get(urlStats,
                            headers: {'Connection': 'Keep-Alive'});
                        final decoded = json.decode(response.body);

                        EasyLoading.show(status: 'En traitement...');
                        int pdt = decoded['n_pdt'];
                        int gt4 = decoded['h > 4po'];
                        int gt17 = decoded['d > 1.7po'];
                        int totalW = decoded['totalWeight'];
                        var heights = decoded['heights'] as List;
                        var diameters = decoded['diameters'] as List;
                        var weights = decoded['weights'] as List;

                        List<num> listHeights = heights.cast<num>();
                        List<num> listDiameters = diameters.cast<num>();
                        List<num> listWeights = weights.cast<num>();

                        // Update the global values
                        globals.totalPDT += pdt;
                        globals.gt4po += gt4;
                        globals.gt17po += gt17;
                        globals.totalWeight += totalW;

                        globals.allDiameters += listDiameters;
                        globals.allHeights += listHeights;
                        globals.allWeights += listWeights;

                        // Compute the percentages
                        globals.gt4pc = (globals.gt4po / globals.totalPDT * 100)
                            .toStringAsFixed(1);
                        globals.gt17pc =
                            (globals.gt17po / globals.totalPDT * 100)
                                .toStringAsFixed(1);

                        // Compute statistics
                        var statsH = Stats.fromData(globals.allHeights);
                        var statsD = Stats.fromData(globals.allDiameters);
                        var statsW = Stats.fromData(globals.allWeights);

                        // Get the means
                        globals.meanHeight = statsH.average.toInt();
                        globals.meanDiameter = statsD.average.toInt();
                        globals.meanWeight = statsW.average.toInt();

                        // Get the median
                        globals.medHeight = statsH.median.toInt();
                        globals.medDiameter = statsD.median.toInt();
                        globals.medWeight = statsW.median.toInt();

                        final calibResp = await http.get(
                            Uri.parse('https://10.0.2.2:5000/calibres'),
                            headers: {'Connection': 'Keep-Alive'});
                        final calibjson = json.decode(calibResp.body);

                        for (var c in globals.calibresMap.keys) {
                          globals.calibresMap[c] += calibjson[c];
                        }

                        EasyLoading.showSuccess('Statistiques mises à jour!');
                        if (mounted) {
                          Navigator.pop(context);
                        }
                      },
                      style: ElevatedButton.styleFrom(
                          backgroundColor:
                              const Color.fromARGB(255, 25, 56, 26)),
                      child: Padding(
                          padding: const EdgeInsets.all(10),
                          child: Text('Accepter',
                              style: GoogleFonts.openSans(fontSize: 20))))
                ],
              )
            ])));
  }
}
