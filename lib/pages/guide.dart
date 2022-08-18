import 'package:flutter/material.dart';

class GuidePage extends StatefulWidget {
  const GuidePage({Key? key}) : super(key: key);

  @override
  State<GuidePage> createState() => _GuidePageState();
}

class _GuidePageState extends State<GuidePage> {
  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
        child: Column(
      children: const [
        ExpansionTile(
          title: Text('Bienvenue!',
              textAlign: TextAlign.left, style: TextStyle(fontSize: 16)),
          subtitle: Text("Contexte et objectifs de l'application"),
          children: <Widget>[
            ListTile(
                title: Text(
                    "Cette application a comme objectif la calibration automatique de pommes de terre. Des informations liées à la longueur, le diamètre et le poids des objets peuvent être obtenus. ")),
          ],
        ),
        ExpansionTile(
          title: Text('Comment prendre de bonnes photos'),
          subtitle: Text('Afin de maximiser la qualité des résultats'),
          children: <Widget>[
            ListTile(
                title: Text(
              '''- Tous les objets doivent être entièrement dans \n   l'image.\n- Évitez un trop grand nombre d'objets dans \n   l'image. Un maximum de 100 est conseillé.\n-	Éliminez les objets superflus de l'image.\n-	Évitez les photos floues.\n-	Évitez un éclairage trop sombre.\n-	Évitez un arrière-plan trop peu contrastant.\n-	Éliminez tout chevauchement entre les objets.\n- Placez au minimum trois cibles dans l'image.\n''',
              textAlign: TextAlign.left,
              style: TextStyle(fontSize: 16),
            )),
          ],
        ),
        ExpansionTile(
          title: Text('Validation des photos'),
          subtitle: Text('Quoi vérifier et comment corriger'),
          children: <Widget>[
            ListTile(
                title: Text(
                    'Lorsque la fenêtre de validation apparaît, vérifier si les cibles (encadrées en vert) et les pommes de terre (marquées d’un X rouge) sont bien localisées.\n\nSi certaines pommes de terre ne sont pas identifiées, vérifiez qu’il n’y a pas de chevauchements ou essayez de les espacer davantage. \n\nSi vous voyez des X rouges là où il n’y a pas de pommes de terre, vérifiez qu’il n’y a pas d’objet superflu ou que l’éclairage est relativement uniforme. \n\nSi une ou plusieurs cibles ne sont pas identifiées, mais qu’il en reste un minimum de deux bien localisées, vous pouvez procéder à la calibration.  Pour corriger des cibles pas ou mal encadrées, vous pouvez tentez de les déplacer. Assurez-vous que l’entièreté de chaque cible soit visible.\n',
                    textAlign: TextAlign.left,
                    style: TextStyle(fontSize: 16))),
          ],
        ),
        ExpansionTile(
          title: Text("Création d'un rapport"),
          subtitle: Text('Informations et trucs'),
          children: <Widget>[
            ListTile(
                title: Text(
                    'Un échantillon volumineux peut être séparés sur plusieurs photos. Par défaut, l’application accumule les résultats et produit un rapport sur toutes les photos. Si vous souhaitez avoir un rapport indépendant pour chaque image, il vous suffit de supprimer les données entre chaque acquisition. \n',
                    textAlign: TextAlign.left,
                    style: TextStyle(fontSize: 16))),
          ],
        ),
        ExpansionTile(
          title: Text("Taille d'échantillon"),
          subtitle: Text("Comment atteindre la marge d'erreur désirée"),
          children: <Widget>[
            ListTile(
                title: Text(
                    "La marge d’erreur diminue lorsque la taille d’échantillon augmente. Voici quelques exemple de marges d’erreur typiques et de la taille d’échantillon requise pour les atteindre* : \n\n  – Marge d'erreur de 5%: 385 \n  – Marge d'erreur de 4%: 601 \n  – Marge d'erreur de 3%: 1068 \n  – Marge d'erreur de 2%: 2401 \n  – Marge d'erreur de 1%: 9604\n *Ces chiffres considèrent un niveau de confiance de 95%.\n",
                    textAlign: TextAlign.left,
                    style: TextStyle(fontSize: 16))),
          ],
        ),
        ExpansionTile(
          title: Text('Autres questions fréquentes'),
          children: <Widget>[
            ListTile(
                title: Text(
                    "Q: L'application me dit que les statistiques sont à jour, mais elles n'apparaissent pas dans la section « Rapport ». Quoi faire?\nR: Il faut rafraîchir la page (bouton « Refresh » en bas à droite).\n",
                    textAlign: TextAlign.left,
                    style: TextStyle(fontSize: 16))),
          ],
        ),
      ],
    ));
  }
}
