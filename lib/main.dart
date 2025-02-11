import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  File? _image;
  String _result = "";
  tfl.Interpreter? _interpreter;
  bool _isModelLoaded = false;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset("assets/yuris_cat_dog_classifier.tflite");
      setState(() {
      _isModelLoaded = true;
      });
      print("✅ TFLite model loaded successfully!");
    } catch (e) {
      print("❌ Erreur lors du chargement du modèle: $e");
    }
  }

  Future<void> classifyImage(Uint8List imageData) async {
    if (_interpreter == null) { // This check if the model interpreter has been loaded if it is null then the model hasn't loaded yet
      print("⚠️ Le modèle n'est pas chargé !");
      return;
    }

    try {
      // Convert Uint8List to Float32 (normalize values between 0 and 1)
      List<List<List<List<double>>>> input = convertImageToModelInput(imageData);

      // Output: Single float value so either a cat or a dog
      var output = List.filled(1, 0.0).reshape([1, 1]);

      _interpreter!.run(input, output);// This runs the model with the processed image as input and writes the result to the output tensor
      // The ! after the _interpreter signifies that it isn't null
      // Sigmoid activation: If output > 0.5, it's a Dog, otherwise it's a Cat
      setState(() {
        _result = output[0][0] > 0.5 ? "Chien 🐶" : "Chat 🐱";
      });
    } catch (e) {
      print("❌ Erreur lors de la classification de l'image: $e"); // If this happens i'll just go punch a wall no joke
    }
  }

  Future<void> pickImage(ImageSource source) async { // Source peut etre caméra ou gallery Également Future represente une valeur qui n'est pas encore accessible mais qui le sera dans le future
    final picker = ImagePicker(); // final == immutable donc elle ne peux changer apres son initialisation
    final pickedFile = await picker.pickImage(source: source); // Attends que le boug prenne une image

    if (pickedFile != null) {
      File imageFile = File(pickedFile.path); // On la converti en image sinon on ne peux pas la lire
      Uint8List processedImage = await preprocessImage(imageFile.path); // On process l'image et on att que la fonction ai fini
      classifyImage(processedImage); // Fait la classification a partir de mon modèle
      setState(() { // Dit a Flutter de refaire L'UI avec l'image
        _image = imageFile; // _image attribut privé qui ne peux etre utilisé que dans cette classe
      });
    }
  }

  Future<Uint8List> preprocessImage(String imagePath) async {
    img.Image? image = img.decodeImage(await File(imagePath).readAsBytes());
    if (image == null) return Uint8List(0);

    // Resize to (150,150) as expected by the model
    img.Image resized = img.copyResize(image, width: 150, height: 150);

    // Convert to Uint8List
    return Uint8List.fromList(img.encodeJpg(resized));
  }

  List<List<List<List<double>>>> convertImageToModelInput(Uint8List imageData) { // This thing returns a 4D tensor in this format [ batch_size (1 because we classify 1 image at a time), height, width, color channels ]

    img.Image? image = img.decodeImage(imageData);
    if (image == null) return [[[[0.0]]]]; // Fallback empty tensor

    List<List<List<List<double>>>> input = List.generate(
      1, // batch size
          (i) => List.generate(
        150, // height
            (y) => List.generate(
          150, //width
              (x) => [ // Color channels
            image.getPixel(x, y).r / 255.0, // Red
            image.getPixel(x, y).g / 255.0, // Green
            image.getPixel(x, y).b / 255.0  // Blue
          ],
        ),
      ),
    );
    return input;
  }


  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    const primaryColor = Color(0xFF0288D1);
    return MaterialApp(
      title: 'Flutter ML',
      theme: new ThemeData(scaffoldBackgroundColor: const Color(0xFFFFFFFF)),
      home: Scaffold(
        appBar: AppBar(title: Text('ML avec Flutter'),
          backgroundColor: primaryColor,
          titleTextStyle: TextStyle(
          color: Color(0xFFFFFFFF) , // Text Color
          fontSize: 20.0,
          ),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _image != null ? Image.file(_image!) : Text("Aucune image sélectionnée"),
              SizedBox(height: 20),
              Text(_result, textAlign: TextAlign.center, style: TextStyle(fontSize: 18)),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () => pickImage(ImageSource.gallery),
                child: Text("Choisir une image"),
              ),
              SizedBox(height: 10),
              ElevatedButton(
                onPressed: () => pickImage(ImageSource.camera),
                child: Text("Prendre une photo"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
