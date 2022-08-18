import 'dart:math';
import 'dart:typed_data';

import 'package:maxiplant/report/report.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:maxiplant/globals.dart' as globals;

Future<Uint8List> makePdf(Report report) async {
  final pdf = Document();
  final imageLogo = MemoryImage(
      (await rootBundle.load('assets/logo.png')).buffer.asUint8List());
  pdf.addPage(
    Page(
      build: (context) {
        return Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  children: [
                    Text("Nom de la calibration: ${report.calibrationName}"),
                    Text("Adresse: ${report.adresse}"),
                    Text("Nom de l'opérateur: ${report.operatorName}"),
                    Text("Date: ${report.date}"),
                  ],
                  crossAxisAlignment: CrossAxisAlignment.start,
                ),
                SizedBox(
                  height: 80,
                  width: 80,
                  child: Image(imageLogo),
                )
              ],
            ),
            SizedBox(height: 20),
            Text('NOMBRE DE POMMES DE TERRE TOTAL: ${report.nPDT}',
                textAlign: TextAlign.left),
            SizedBox(height: 20),
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Column(children: [
                PaddedText('1. Dimensions'),
                SizedBox(
                    width: 200,
                    height: 200,
                    child: Table(
                      border: TableBorder.all(color: PdfColors.black),
                      children: [
                        TableRow(
                          children: [
                            PaddedText('Caractéristique',
                                align: TextAlign.right),
                            PaddedText('Résultat (mm)',
                                align: TextAlign.center),
                          ],
                        ),
                        TableRow(
                          children: [
                            PaddedText('Longueur moyenne',
                                align: TextAlign.right),
                            PaddedText('${report.meanH} mm',
                                align: TextAlign.center),
                          ],
                        ),
                        TableRow(
                          children: [
                            PaddedText('Longueur médiane',
                                align: TextAlign.right),
                            PaddedText('${report.medH} mm',
                                align: TextAlign.center),
                          ],
                        ),
                        TableRow(
                          children: [
                            PaddedText('Largeur moyenne',
                                align: TextAlign.right),
                            PaddedText('${report.meanD} mm',
                                align: TextAlign.center),
                          ],
                        ),
                        TableRow(
                          children: [
                            PaddedText('Largueur médiane',
                                align: TextAlign.right),
                            PaddedText('${report.medD} mm',
                                align: TextAlign.center),
                          ],
                        )
                      ],
                    ))
              ]),
              Container(width: 50),
              Column(children: [
                PaddedText('2. Poids'),
                SizedBox(
                    width: 200,
                    height: 200,
                    child: Table(
                      border: TableBorder.all(color: PdfColors.black),
                      children: [
                        TableRow(
                          children: [
                            PaddedText('Caractéristique',
                                align: TextAlign.right),
                            PaddedText('Résultat (g)', align: TextAlign.center),
                          ],
                        ),
                        TableRow(
                          children: [
                            PaddedText('Poids moyen', align: TextAlign.right),
                            PaddedText('${report.meanW}',
                                align: TextAlign.center),
                          ],
                        ),
                        TableRow(
                          children: [
                            PaddedText('Poids médian', align: TextAlign.right),
                            PaddedText('${report.medW}',
                                align: TextAlign.center),
                          ],
                        ),
                      ],
                    )),
                SizedBox(height: 20),
                SizedBox(
                    width: 200,
                    height: 200,
                    child: Table(
                      border: TableBorder.all(color: PdfColors.black),
                      children: [
                        TableRow(
                          children: [
                            PaddedText('Poids total (kg):',
                                align: TextAlign.right),
                            PaddedText(
                                '${(report.totalW / 1000).toStringAsFixed(1)} +/- ${(report.totalW / 1000 * 0.02).toStringAsFixed(1)}',
                                align: TextAlign.center),
                          ],
                        )
                      ],
                    ))
              ])
            ]),
            SizedBox(height: 30),
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Column(children: [
                PaddedText('3. Calibres'),
                Text(
                    "Marge d'erreur: ${(98 / sqrt(globals.totalPDT)).toStringAsFixed(1)} %"),
                SizedBox(height: 5),
                SizedBox(
                    width: 200,
                    height: 200,
                    child: Table(
                      border: TableBorder.all(color: PdfColors.black),
                      children: [
                            TableRow(
                              children: [
                                PaddedText('Calibre', align: TextAlign.right),
                                PaddedText('Nombre', align: TextAlign.center),
                                PaddedText('%', align: TextAlign.center),
                              ],
                            )
                          ] +
                          [
                            for (var entry in globals.calibresMap.entries)
                              TableRow(
                                children: [
                                  PaddedText('${entry.key.substring(2)}',
                                      align: TextAlign.right),
                                  PaddedText('${entry.value}',
                                      align: TextAlign.center),
                                  PaddedText(
                                      '${(entry.value / report.nPDT * 100).toStringAsFixed(1)}',
                                      align: TextAlign.center),
                                ],
                              )
                          ],
                    ))
              ]),
              Container(width: 50),
              Column(children: [
                PaddedText('4. Statistiques spécifiques'),
                Text(
                    "Marge d'erreur: ${(98 / sqrt(globals.totalPDT)).toStringAsFixed(1)} %"),
                SizedBox(height: 5),
                SizedBox(
                    width: 200,
                    height: 200,
                    child: Table(
                      border: TableBorder.all(color: PdfColors.black),
                      children: [
                        TableRow(
                          children: [
                            PaddedText('Caractéristique',
                                align: TextAlign.right),
                            PaddedText('Nombre', align: TextAlign.center),
                            PaddedText('%', align: TextAlign.center),
                          ],
                        ),
                        TableRow(
                          children: [
                            PaddedText('Diamètre > 1.7 po',
                                align: TextAlign.right),
                            PaddedText('${report.gt17}',
                                align: TextAlign.center),
                            PaddedText(report.gt17p, align: TextAlign.center),
                          ],
                        ),
                        TableRow(
                          children: [
                            PaddedText('Longueur > 4 po',
                                align: TextAlign.right),
                            PaddedText('${report.gt4}',
                                align: TextAlign.center),
                            PaddedText(report.gt4p, align: TextAlign.center),
                          ],
                        ),
                      ],
                    ))
              ])
            ]),
            SizedBox(height: 20),
            Divider(
              height: 1,
              borderStyle: BorderStyle.dashed,
            ),
            SizedBox(height: 10),
            Container(
              width: 550,
              height: 120,
              margin: const EdgeInsets.all(15.0),
              padding: const EdgeInsets.all(10.0),
              decoration: BoxDecoration(border: Border.all()),
              child: Text('Commentaires: ${report.comment}'),
            )
          ],
        );
      },
    ),
  );
  return pdf.save();
}

// ignore: non_constant_identifier_names
Widget PaddedText(
  final String text, {
  final TextAlign align = TextAlign.left,
}) =>
    Padding(
      padding: const EdgeInsets.all(5),
      child: Text(
        text,
        textAlign: align,
      ),
    );
