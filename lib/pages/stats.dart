import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:maxiplant/globals.dart' as globals;
import 'package:maxiplant/routes/dialogs.dart';

class StatsPage extends StatefulWidget {
  const StatsPage({
    Key? key,
  }) : super(key: key);

  @override
  State<StatsPage> createState() => _StatsPageState();
}

class _StatsPageState extends State<StatsPage> {
  int pdt = globals.totalPDT;
  int meanH = globals.meanHeight;
  int meanD = globals.meanDiameter;
  int medH = globals.medHeight;
  int medD = globals.medDiameter;
  int gt4 = globals.gt4po;
  int gt17 = globals.gt17po;
  String gt17p = globals.gt17pc;
  String gt4p = globals.gt4pc;
  int meanW = globals.meanWeight;
  int medW = globals.medWeight;
  int totalW = globals.totalWeight;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: SingleChildScrollView(
          child: Container(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Nombres de pommes de terre: $pdt',
                          style: GoogleFonts.courierPrime(fontSize: 18)),
                      const SizedBox(height: 10),
                      Text(
                          'Longueur\n   Moyenne: $meanH mm\n   Médiane: $medH mm\n   Sup. à 4po: $gt4 ($gt4p %)',
                          style: GoogleFonts.courierPrime(fontSize: 18)),
                      const SizedBox(height: 10),
                      Text(
                          'Diamètre\n   Moyen: $meanD mm\n   Médian: $medD mm\n   Sup. à 1.7po: $gt17 ($gt17p %)',
                          style: GoogleFonts.courierPrime(fontSize: 18)),
                      const SizedBox(height: 10),
                      Text(
                          'Poids\n   Moyen: $meanW  g\n   Médian: $medW  g\n   Total: $totalW g',
                          style: GoogleFonts.courierPrime(fontSize: 18)),
                      const SizedBox(height: 10),
                      Text('Calibres',
                          style: GoogleFonts.courierPrime(fontSize: 18)),
                    ],
                  ),
                  Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        for (var entry in globals.calibresMap.entries)
                          Text('  ${entry.key.substring(2)}: ${entry.value}',
                              style: GoogleFonts.courierPrime(fontSize: 18))
                      ]),
                  const SizedBox(height: 80),
                  Row(children: [
                    const SizedBox(width: 15),
                    ElevatedButton(
                      onPressed: () {
                        showDialogWithFields(context);
                      },
                      style: ElevatedButton.styleFrom(
                          textStyle: const TextStyle(fontSize: 14)),
                      child: Padding(
                          padding: const EdgeInsets.all(5),
                          child: Text('Exporter le\nrapport',
                              textAlign: TextAlign.center,
                              style: GoogleFonts.openSans(fontSize: 14))),
                    ),
                    const SizedBox(width: 20),
                    ElevatedButton(
                        onPressed: () {
                          showAlertDialog(context);
                        },
                        style: ElevatedButton.styleFrom(
                            backgroundColor:
                                const Color.fromARGB(255, 130, 26, 26)),
                        child: Padding(
                            padding: const EdgeInsets.all(5),
                            child: Text('Supprimer \nles données',
                                textAlign: TextAlign.center,
                                style: GoogleFonts.openSans(fontSize: 14))))
                  ])
                ],
              ))),
      floatingActionButton: Padding(
          padding: const EdgeInsets.only(bottom: 5.0, right: 12.0),
          child: FloatingActionButton(
            heroTag: 'refresh',
            onPressed: () {
              setState(() {
                pdt = globals.totalPDT;
                meanH = globals.meanHeight;
                meanD = globals.meanDiameter;
                medH = globals.medHeight;
                medD = globals.medDiameter;
                gt4 = globals.gt4po;
                gt17 = globals.gt17po;
                gt17p = globals.gt17pc;
                gt4p = globals.gt4pc;
                meanW = globals.meanWeight;
                medW = globals.medWeight;
                totalW = globals.totalWeight;
              });
            },
            backgroundColor: Colors.green,
            child: const Icon(
              Icons.refresh,
              size: 40,
            ),
          )),
    );
  }
}
