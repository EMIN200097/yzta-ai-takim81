import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import 'sitting.dart';
import 'home.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WebSocket Camera Stream',
      home: HomePage()
    );
  }
}