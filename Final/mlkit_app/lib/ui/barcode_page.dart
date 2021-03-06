import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:image_picker/image_picker.dart';
import 'package:firebase_ml_vision/firebase_ml_vision.dart';
import 'dart:io';

class BarcodePage extends StatefulWidget {
  @override
  _BarcodePageState createState() => _BarcodePageState();
}

class _BarcodePageState extends State<BarcodePage> {

  final GlobalKey<ScaffoldState> _scaffoldKey = new GlobalKey<ScaffoldState>();

  File _image;
  final picker = ImagePicker();

  String textBarCode = '';

  final Widget takeImage = SvgPicture.asset(
      'assets/image.svg',
      semanticsLabel: 'Acme Logo'
  );

  Widget photoButton() {
    return Container(
      margin: EdgeInsets.only(top: 10.0, bottom: 10.0, left: 50.0, right: 50.0),
      width: MediaQuery.of(context).size.width,
      height: 46.0,
      child: GestureDetector(
        onTap: (){
          selectImage(_scaffoldKey.currentContext);
        },
        child: Container(
            child: Center(
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  Icon(Icons.camera_alt, color: Colors.white,),
                  Container(width: 6,),
                  Text(
                    'Analizar Foto',
                    style: TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 18.0,
                        color: Colors.white
                    ),
                  ),
                ],
              ),
            ),
            decoration: BoxDecoration(
                color: Colors.amber,
                borderRadius: BorderRadius.all(Radius.circular(12.0))
            )
        ),
      ),
    );
  }

  void selectImage(context) {
    showModalBottomSheet(
        context: context,
        builder: (BuildContext builder) {
          return Container(
              color: Colors.white,
              child: Container(
                child: new Wrap(
                  children: <Widget>[
                    new Center(
                      child: ListTile(
                          leading: new Icon(Icons.camera_alt),
                          title: new Text('Tomar Foto'),
                          onTap: () {
                            Navigator.pop(context);
                            fromCamera();
                          }),
                    ),
                    new Center(
                      child: ListTile(
                          leading: new Icon(Icons.photo),
                          title: new Text('Seleccionar Imagen'),
                          onTap: () {
                            Navigator.pop(context);
                            fromGallery();
                          }),
                    ),
                  ],
                ),
              )
          );
        }
    );
  }

  void fromCamera() async{
    print('Camara');
    //final pickedFile = await picker.getImage(source: ImageSource.camera);
    var image = await ImagePicker.pickImage(source: ImageSource.camera);
    setState(() {
      //_image = File(pickedFile.path);
      _image = image;
    });
    _initializeVision();
  }

  void fromGallery() async{
    print('Galeria');
    var image = await ImagePicker.pickImage(source: ImageSource.gallery);
    setState(() {
      _image = image;
    });
    _initializeVision();
  }

  void _initializeVision() async {
    final FirebaseVisionImage visionImage = FirebaseVisionImage.fromFile(_image);
    final BarcodeDetector barcodeDetector = FirebaseVision.instance.barcodeDetector();
    final List<Barcode> barCodes = await barcodeDetector.detectInImage(visionImage);
    for (Barcode barcode in barCodes) {
      final String rawValue = barcode.rawValue;
      final BarcodeValueType valueType = barcode.valueType;
      setState(() {
        textBarCode = "$rawValue\nType: $valueType";
      });}
  }


  Widget bodyWidget(){
    return Container(
      width: double.infinity,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Container(
            margin: EdgeInsets.only(top: 30, bottom: 10),
            width: 300,
            height: 400,
            child: _image == null? takeImage: Image.file(_image, fit: BoxFit.fitWidth,),
          ),
          photoButton(),
          Container(
            margin: EdgeInsets.only(top: 10, bottom: 30, left: 50, right: 50),
            child: Text(textBarCode),
          )
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        key: _scaffoldKey,
        appBar: AppBar(
          title: Text('Barcode Detector'),
          centerTitle: true,
        ),
        body: SingleChildScrollView(
          scrollDirection: Axis.vertical,
          child: bodyWidget(),
        )
    );
  }
}
