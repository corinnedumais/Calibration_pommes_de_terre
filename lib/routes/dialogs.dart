import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:maxiplant/report/report.dart';
import 'package:maxiplant/globals.dart' as globals;
import 'package:maxiplant/routes/pdfpreview.dart';

showAlertDialog(BuildContext context) {
  // set up the buttons
  Widget confirmButton = TextButton(
    style: TextButton.styleFrom(
      foregroundColor: Colors.blue,
    ),
    onPressed: () {
      globals.totalPDT = 0;
      globals.meanHeight = 0;
      globals.meanDiameter = 0;
      globals.medHeight = 0;
      globals.medDiameter = 0;
      globals.calibresMap = {
        'a) 3 po+': 0,
        'b) 3 po': 0,
        'c) 2 1/2 po': 0,
        'd) 2 1/4 po': 0,
        'e) 2 po': 0,
        'f) 1 7/8 po': 0,
        'g) 1 3/4 po': 0
      };
      globals.gt4po = 0;
      globals.gt17po = 0;
      globals.gt4pc = '';
      globals.gt17pc = '';
      globals.meanWeight = 0;
      globals.medWeight = 0;
      globals.totalWeight = 0;
      Navigator.of(context).pop();
    },
    child: const Text("Confirmer"),
  );
  Widget cancelButton = TextButton(
    style: TextButton.styleFrom(
      foregroundColor: Colors.blue,
    ),
    child: const Text("Annuler"),
    onPressed: () {
      Navigator.of(context).pop();
    },
  );
  // set up the AlertDialog
  AlertDialog alert = AlertDialog(
    title: const Text("Avertissement"),
    actionsAlignment: MainAxisAlignment.center,
    content: const Text(
        "Cette action est sur le point de supprimer toutes les données.\n\nEst-ce vraiment ce que vous souhaitez faire?"),
    actions: [
      confirmButton,
      cancelButton,
    ],
  ); // show the dialog
  showDialog(
    context: context,
    builder: (BuildContext context) {
      return alert;
    },
  );
}

Future showDialogWithFields(BuildContext context) {
  var idController = TextEditingController();
  var operatorController = TextEditingController();
  var adressController = TextEditingController();
  var messageController = TextEditingController();

  return showDialog(
    context: context,
    barrierDismissible: false,
    builder: (BuildContext context) {
      return AlertDialog(
        scrollable: true,
        title: const Text('Informations'),
        content: SizedBox(
            height: 300,
            child: Column(
              children: [
                Expanded(
                    child: TextFormField(
                  controller: idController,
                  decoration:
                      const InputDecoration(hintText: 'Numéro de calibration'),
                )),
                Expanded(
                    child: TextFormField(
                  controller: adressController,
                  decoration: const InputDecoration(hintText: 'Adresse'),
                )),
                Expanded(
                    child: TextFormField(
                  controller: operatorController,
                  decoration:
                      const InputDecoration(hintText: "Nom de l'opérateur"),
                )),
                Expanded(
                    child: TextFormField(
                  controller: messageController,
                  decoration: const InputDecoration(hintText: 'Commentaires'),
                ))
              ],
            )),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Annuler'),
          ),
          TextButton(
            onPressed: () {
              // Send them to your email maybe?
              var operator = operatorController.text;
              var id = idController.text;
              var adress = adressController.text;
              var message = messageController.text;

              DateTime now = DateTime.now();
              String formattedDate = DateFormat('yyyy-MM-dd kk:mm').format(now);

              // Create Rapport instance
              Report report = Report(
                  calibrationName: id,
                  operatorName: operator,
                  adresse: adress,
                  comment: message,
                  date: formattedDate,
                  nPDT: globals.totalPDT,
                  meanH: globals.meanHeight,
                  meanD: globals.meanDiameter,
                  meanW: globals.meanWeight,
                  medH: globals.medHeight,
                  medD: globals.medDiameter,
                  medW: globals.medWeight,
                  totalW: globals.totalWeight,
                  gt4: globals.gt4po,
                  gt17: globals.gt17po,
                  gt4p: globals.gt4pc,
                  gt17p: globals.gt17pc,
                  calibres: globals.calibresMap);

              Navigator.of(context).push(MaterialPageRoute(
                  builder: (context) => PdfPreviewPage(rapport: report)));
            },
            child: const Text('Créer le PDF'),
          ),
        ],
      );
    },
  );
}
