import 'dart:io';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:maxiplant/routes/validcalib.dart';

class CalibratePage extends StatefulWidget {
  const CalibratePage({Key? key}) : super(key: key);

  @override
  State<CalibratePage> createState() => _CalibratePageState();
}

class _CalibratePageState extends State<CalibratePage> {
  File? image;
  String? imageStr;
  String dropDownValue = 'Burbank';

  Future pickImage() async {
    try {
      final image = await ImagePicker().pickImage(source: ImageSource.gallery);

      if (image == null) return;

      final imageTemp = File(image.path);

      setState(() => this.image = imageTemp);
    } on PlatformException catch (e) {
      if (kDebugMode) {
        print('Failed to pick image: $e');
      }
    }
  }

  Future pickImageC() async {
    try {
      final image = await ImagePicker().pickImage(source: ImageSource.camera);

      if (image == null) return;

      final imageTemp = File(image.path);

      setState(() => this.image = imageTemp);
    } on PlatformException catch (e) {
      if (kDebugMode) {
        print('Failed to pick image: $e');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Container(
            height: 800,
            alignment: Alignment.center,
            padding: const EdgeInsets.all(20),
            child: Column(children: [
              const SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton.icon(
                      label: const Text("Gallerie",
                          style: TextStyle(color: Colors.white, fontSize: 18)),
                      onPressed: () {
                        pickImage();
                      },
                      icon: const Icon(Icons.photo_library_sharp)),
                  const SizedBox(width: 40),
                  ElevatedButton.icon(
                      label: const Text("Caméra",
                          style: TextStyle(color: Colors.white, fontSize: 18)),
                      onPressed: () {
                        pickImageC();
                      },
                      icon: const Icon(Icons.photo_camera_sharp)),
                ],
              ),
              const SizedBox(height: 30),
              // ignore: sized_box_for_whitespace
              Container(
                  width: 150,
                  height: 55,
                  child: InputDecorator(
                    decoration:
                        const InputDecoration(border: OutlineInputBorder()),
                    child: DropdownButtonHideUnderline(
                        child: DropdownButton<String>(
                      isExpanded: true,
                      value: dropDownValue,
                      elevation: 16,
                      style: const TextStyle(
                          color: Color.fromARGB(255, 51, 49, 49)),
                      onChanged: (String? newValue) {
                        setState(() {
                          dropDownValue = newValue!;
                        });
                      },
                      items: <String>['Burbank', 'Mountain Gem']
                          .map<DropdownMenuItem<String>>((String value) {
                        return DropdownMenuItem<String>(
                          value: value,
                          child: Center(
                              child: Text(value, textAlign: TextAlign.center)),
                        );
                      }).toList(),
                    )),
                  )),
              const SizedBox(height: 30),
              image != null
                  ? Image.file(image!,
                      width: 400, height: 300, fit: BoxFit.cover)
                  : const Text("Aucune image choisie"),
              const SizedBox(height: 20),
            ])),
        floatingActionButton: Padding(
            padding: const EdgeInsets.only(bottom: 40.0),
            child: FloatingActionButton.extended(
              heroTag: 'calib',
              onPressed: () async {
                if (image == null) {
                  return;
                }

                // Prep image to send it in base64 format
                EasyLoading.show(status: 'Calibration en cours...');
                List<int> imageBytes = image!.readAsBytesSync();
                String base64img = base64.encode(imageBytes);

                var client = http.Client();
                var url = Uri.parse('https://10.0.2.2:5000/calibrer');

                try {
                  // Send image to API through POST request
                  // ignore: unused_local_variable
                  final queryImg = await http.post(url,
                      body: json.encode(
                          {'base64img': base64img, 'variety': dropDownValue}));

                  // Get response through GET request
                  final response = await http
                      .get(url, headers: {'Connection': 'Keep-Alive'});

                  final decoded = json.decode(response.body);

                  // Reconstruct image from json
                  setState(() {
                    imageStr = decoded['img'].toString();
                  });

                  EasyLoading.showSuccess('Terminé!');
                  EasyLoading.dismiss();
                  // ignore: use_build_context_synchronously
                  Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (context) => ValidationRoute(
                                imgStr: imageStr!,
                                pdt: decoded['n_pdt'],
                                targets: decoded['n_targets'],
                              )));
                } finally {
                  client.close();
                }
              },
              label: const Text('Calibrer', style: TextStyle(fontSize: 25)),
              elevation: 15,
            )),
        floatingActionButtonLocation:
            FloatingActionButtonLocation.miniCenterFloat);
  }
}
