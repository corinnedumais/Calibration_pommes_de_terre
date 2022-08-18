import 'package:flutter/material.dart';
import 'package:maxiplant/report/report.dart';
import 'package:printing/printing.dart';
import '../report/pdfexport.dart';

class PdfPreviewPage extends StatelessWidget {
  final Report rapport;
  const PdfPreviewPage({Key? key, required this.rapport}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('PDF Preview'),
      ),
      body: PdfPreview(
        build: (context) => makePdf(rapport),
      ),
    );
  }
}
