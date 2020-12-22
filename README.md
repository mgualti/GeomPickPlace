# GeomPickPlace

Source code for the paper "Pick-and-Place With Uncertain Object Instance Segmentation and Shape Completion"

* **Authors:** Marcus Gualteri
* **Version:** 1.0.0

## Prerequisites

Tensorflow 2.1.0 (https://www.tensorflow.org/install), PointCloudsPython (https://github.com/mgualti/PointCloudsPython), and OpenRAVE (https://github.com/rdiankov/openrave) are required. These instructions for installing OpenRAVE may be helpful: https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html. Matlab is used for plotting results, but this is not a strict requirement. The code was tested on Ubuntu 16.04.

## Dataset generation

First, download and extract 3DNet Cat200 Model Database (https://strands.readthedocs.io/en/latest/datasets/three_d_net.html). Download ShapeNETCore V2 for the bottle arrangement task, since it has more bottles than 3DNet (https://www.shapenet.org/). Use the scripts view_meshes_3dnet.py, view_meshes_shapenet.py, and extract_meshes_shapenet.py to extract and examine the meshes. Partition the meshes into train and test sets as indicated below. (TODO: The ShapeNET object numbers may be platform specific.)

After this, run python/generate_full_clouds_*.py from the Simulation directory. After that, run generate_completions_*.py, generate_segmentations_*.py, generate_grasps_*.py, and generate_places_*.py. This latter step takes a lot of time and space, but it can be skipped if using the pretrained network models from Simulation/tensorflow/models.

Bottles Train:
102.obj 159.obj 213.obj 262.obj 30.obj 365.obj 419.obj 468.obj 61.obj 104.obj 15.obj 214.obj 263.obj 310.obj 367.obj 420.obj 469.obj 62.obj 105.obj 160.obj 216.obj 264.obj 311.obj 368.obj 422.obj 470.obj 63.obj 107.obj 161.obj 217.obj  266.obj 313.obj 369.obj 424.obj 472.obj 65.obj 108.obj 165.obj 218.obj 267.obj 315.obj 36.obj 425.obj 473.obj 67.obj 109.obj  166.obj  21.obj   268.obj  317.obj  371.obj  426.obj  474.obj  68.obj 10.obj   167.obj  221.obj  26.obj   318.obj  372.obj  428.obj  475.obj  6.obj 112.obj  171.obj  222.obj  270.obj  31.obj 374.obj  429.obj  476.obj  70.obj 113.obj  172.obj  224.obj  271.obj  320.obj  376.obj  430.obj  477.obj  71.obj 115.obj  173.obj  226.obj  273.obj  321.obj  377.obj  432.obj  478.obj  72.obj 117.obj  175.obj  227.obj  275.obj  322.obj  379.obj  434.obj  479.obj  75.obj 118.obj  176.obj  228.obj  276.obj  326.obj  383.obj  435.obj  47.obj   77.obj 119.obj  178.obj  22.obj   277.obj  327.obj  384.obj  437.obj  480.obj  78.obj 11.obj   17.obj   230.obj  279.obj  328.obj  385.obj  438.obj  481.obj  7.obj 121.obj  180.obj  231.obj  280.obj  331.obj  387.obj  439.obj  482.obj  80.obj 122.obj  181.obj  233.obj  281.obj  332.obj  388.obj  43.obj   483.obj  81.obj 123.obj  182.obj  235.obj  283.obj  333.obj  38.obj   441.obj  484.obj  82.obj 126.obj  184.obj  236.obj  284.obj  336.obj  390.obj  442.obj  485.obj  84.obj 127.obj  186.obj  237.obj  285.obj  339.obj  393.obj  444.obj  486.obj  85.obj 128.obj  187.obj  239.obj  287.obj  33.obj   394.obj  447.obj  488.obj  86.obj 131.obj  189.obj  240.obj  288.obj  340.obj  395.obj  448.obj  48.obj   88.obj 132.obj  190.obj  241.obj  289.obj  342.obj  399.obj  449.obj  492.obj  89.obj 133.obj  191.obj  243.obj  28.obj   343.obj  39.obj   44.obj   493.obj  90.obj 135.obj  194.obj  245.obj  293.obj  345.obj  3.obj    451.obj  494.obj  93.obj 138.obj  195.obj  246.obj  295.obj  348.obj  400.obj  452.obj  495.obj  94.obj 139.obj  196.obj  249.obj  296.obj  349.obj  402.obj  453.obj  496.obj  95.obj 141.obj  198.obj  24.obj   298.obj  34.obj   405.obj  455.obj  497.obj  97.obj 143.obj  199.obj  250.obj  299.obj  350.obj  406.obj  456.obj  498.obj  98.obj 144.obj  201.obj  251.obj  2.obj    352.obj  407.obj  457.obj  4.obj    99.obj 148.obj  202.obj  253.obj  300.obj  353.obj  40.obj   459.obj  50.obj   9.obj 149.obj  203.obj  255.obj  302.obj  354.obj  410.obj  45.obj   52.obj 14.obj   206.obj  256.obj  303.obj  359.obj  411.obj  461.obj  54.obj 151.obj  207.obj  258.obj  304.obj  360.obj  413.obj  462.obj  55.obj 153.obj  208.obj  259.obj  306.obj  361.obj  415.obj  464.obj  57.obj 155.obj  20.obj   25.obj   307.obj  363.obj  416.obj  465.obj  58.obj 156.obj  211.obj  260.obj  308.obj  364.obj  417.obj  466.obj  59.obj

Bottles Test:
101.obj 158.obj 205.obj 257.obj 305.obj 362.obj 418.obj 467.obj 87.obj 106.obj 163.obj 209.obj 261.obj 309.obj 366.obj  41.obj 46.obj 8.obj 110.obj 168.obj 215.obj 265.obj 314.obj 370.obj 423.obj 471.obj 91.obj 116.obj 174.obj 220.obj 269.obj 319.obj 375.obj 427.obj 51.obj 96.obj 120.obj 179.obj 225.obj 274.obj 325.obj 37.obj 431.obj 56.obj 125.obj 183.obj 229.obj 278.obj 329.obj 382.obj 436.obj 5.obj 130.obj 188.obj 234.obj 27.obj   32.obj   386.obj  440.obj  60.obj 134.obj  193.obj  238.obj  282.obj  334.obj  391.obj  446.obj  64.obj 13.obj   197.obj  23.obj   286.obj  341.obj  397.obj  450.obj  69.obj 140.obj  19.obj   242.obj  291.obj  347.obj  404.obj  454.obj  74.obj 146.obj  1.obj    247.obj  297.obj  351.obj  409.obj  458.obj  79.obj 152.obj  200.obj  252.obj  301.obj  355.obj  414.obj  463.obj  83.obj

Packing Train boat:
12a01b67cb987d385859fb379730f7f7.ply  a0e9a2dd6731eceeaadb7997f5aa04c9.ply 426980cdb3696492dec0b61e1edd7faa.ply  b00e1af5a8d0674efe9d6d96542b8ef4.ply 4af786ed4226705279863338881ed398.ply  bb7ab7697592a2137142d17e95e3ffc9.ply 51537c56f71ec82acfd826dd468a5497.ply  bf2563524de3aa039b1588a265e9bd25.ply 54daf1472c51bb47a97a590141e0046.ply   c7a70182f0b2f8db96929192c76f0ac.ply 5fa144c3a8cb5234379339ae6512a12.ply   cfb7ca78b0b65c4f2d615e80824301ca.ply 6556015faea5ba45e0f0f0669675011.ply   e3206eef4da407e7c08fee43ebed0bfa.ply 6aa5d719f1c159c0991df8a7bda22835.ply  edb4574369f95b50689cbb453f479f9f.ply 6af167d3c7c06b2869416124f6942827.ply  eff7e84e2098a6fd37763367b6df21ae.ply 87a750e55b04395e668e250d23d1f86a.ply  fbd9f050c1ddbb9871e397fe45dce6b.ply 9b02ecc129d89000f4841afd366e16cb.ply  fe7362e8a02d00072e4aadf908a27d12.ply 9f34a30515f934d4ec56a2dfc7a61313.ply

Packing Train bottle:
109d55a137c042f5760315ac3bf2c13e.ply 9f2bb4a157164af19a7c9976093a710d.ply 114509277e76e413c8724d5673a063a6.ply  a1bc36109cd382b78340c8802f48c170.ply 15787789482f045d8add95bf56d3d2fa.ply  a429f8eb0c3e6a1e6ea2d79f658bbae7.ply 1d4480abe9aa45ce51a99c0e19a8a54.ply   a86d587f38569fdf394a7890920ef7fd.ply 1ee865178cd126fe3bfd7d6656f05358.ply  a87fc2164d5bb73b9a6e43b878d5b335.ply 1ef68777bfdb7d6ba7a07ee616e34cd7.ply  aa868402ff149def8552149121505df9.ply 20b7adb178ea2c71d8892a9c05c4aa0e.ply  acc67510dd228d01d7cbb1f9251b6139.ply 3432ee42351384cff4bdca145ed48dc6.ply  b0652a09588f293c7e95755f464f6241.ply 3dbd66422997d234b811ffed11682339.ply  bf7ecd80f7c419feca972daa503b3095.ply 437678d4bc6be981c8724d5673a063a6.ply  c13219fac28e722edd6a2f6a8ecad52d.ply 46b5318e39afe48a30eaaf40a8a562c1.ply  c4729e522fd3d750def51fa1c8b9ff22.ply 490f3d001d230e6889f687b6e87e144f.ply  c5e425b9b1f4f42b6d7d15cb5e1928e.ply 4b5f54fc4e629371cf078dc7b29022e6.ply  d44472ef7086538676bb31db0358e9c6.ply 5ad47181a9026fc728cc22dce7529b69.ply  d851cbc873de1c4d3b6eb309177a6753.ply 5e67380e99a257814e5292217a77cd7.ply   d8b6c270d29c58c55627157b31e16dc2.ply 642092b63e83ac4dd01d3c0e394d0302.ply  d9aee510fd5e8afb93fb5c975e8de2b7.ply 6623907ab044311af4bdca145ed48dc6.ply  da2703e6d87a28e75887f1f81e7530ec.ply 684ff2b770c26616d3dfba73f54d35bb.ply  dacc6638cd62d82f42ebc0504c999b.ply 6b810dbc89542fd8a531220b48579115.ply  dc687759ea93d1b72cd6cd3dc3fb5dc2.ply 6b8b2cb01c376064c8724d5673a063a6.ply  defc45107217afb846564a8a219239b.ply 70e77c09aca88d4cf76fe74de9d44699.ply  e04e680664e616037199c3a1b4ff8300.ply 736c26e42954ecfecab7ab7d91fa9137.ply  e7714b3280fa7f925b2cc658332a8e03.ply 7467b9892496a83fbf8b9d530de7a108.ply  e8b48d395d3d8744e53e6e0633163da8.ply 81d289cf00a544f62d9fe390e23fd20f.ply  eae094f41e2cf47461af36d49e6d6f51.ply 8309e710832c07f91082f2ea630bf69e.ply  ee74f5bfb0d7c8a5bd288303be3d57e7.ply 8cd9b10f611ac28e866a1445c8fba9da.ply  f452c1053f88cd2fc21f7907838a35d1.ply 908e85e13c6fbde0a1ca08763d503f0e.ply  fa44223c6f785c60e71da2487cb2ee5b.ply 970027a2c9f1de88c123147a914238ea.ply  ketchup_bottle.ply

Packing train box:
1dd407598b5850959b1500745a428d00.ply  85286a1d1c7a241ac77fa524ced99227.ply 33d55b2e524b3267e1caf415acc56c13.ply  9b150eac28e668f36776771c8bc4ff9.ply 37558c8a9586d6e08f9ce7abd35d923.ply   a9beebe1b851adacd59ff053d1480d84.ply 3e054a6106748dad4b54bd173e667d97.ply  abc86fa217ffeeeab3c769e22341ffb4.ply 5062771877ee2a229eeafe87febc2d14.ply  c87bb65a628e0ca6a4bd57b754f4233c.ply 55c78236eb6092cfa4aa3c4ef0a792a8.ply  cf901a04f607b06a4f156e2e3267f52d.ply 746be0b997f0500f3752a5ea854be89d.ply  d50f26555d4efc1a36edb1220f900995.ply 77b7c88d0ce53dc976c07cb3ca5f9091.ply  e481c3d08c11fc9e2978181091f6882a.ply 77f28122cb38e9b4bf844b066ff38d44.ply  f822931fe07a7654ce41b639931f9ca1.ply 7d3f27ed99f3e3c13ee5fe8e4bef5c3f.ply

Packing train car:
15b094d1b52586e77d2ee99b69c27298.ply  7f07a941954fb9ea70a43c2d978e502e.ply 198d8e274c59511b36a1a31af8f59.ply     8480c345c555a00aaa69dfdc5532bb13.ply 1ac489c3461e09f07b882d1dc650f6f7.ply  85f6145747a203becc08ff8f1f541268.ply 1acfbda4ce0ec524bedced414fad522f.ply  864c3c85367a304e7302a5e36ce363de.ply 22a7dd59bd38ca843343f5fadc4a2136.ply  89026c748b9cf721bda72093f9b5aa73.ply 2f3a1e4c55dcdbe2f57aaf653ef4f8f7.ply  99774add173045e844608b3255ca1886.ply 30effd773a9bef103d4e35ba1985c7c2.ply  a08b151ac5254df184bf231ea6363fab.ply 33568ae2a3a39c0715d4d2fdca08573e.ply  a2b57a3a8a9bf50e723552943cef3fcd.ply 369c8878ed7114a462bc404553f4c03e.ply  ab129605a0303bfd1ab9df4be75138d0.ply 3a3f3ee1eb027cdca413908c0e169330.ply  b5a36bdcea489a836f937c53d01bac7.ply 3c7d2c8bbd0c1ee5a413908c0e169330.ply  bb1afdd76bd9555abf3af70ab74680f8.ply 41cbc4013776e425ee03f96086e8207f.ply  bd2b0b61449846cb13d1db0bebe167bc.ply 41cdb617e0ae08970508d66092e983e.ply   beedf0f34425a8229ad0cf86a19e7895.ply 42134787c805fffe1ab9df4be75138d0.ply  c15f0c5b1e902251426b54880606cd86.ply 4c5c32eaaf1fc80aa413908c0e169330.ply  cce59fb77e500b99d3f267930fa9351c.ply 4cce557de0c31a0e70a43c2d978e502e.ply  ce77b7e28b298ce21ab9df4be75138d0.ply 5343e944a7753108aa69dfdc5532bb13.ply  d2428faf5fccfd4d84688264755620b0.ply 5555f8433d2bf125dce1a6514d4f1ab5.ply  e76471d97a1ffbc97302a5e36ce363de.ply 6291f064e64c6eb9d18a5bd35f0d115b.ply  ecb93a0119c1cdd36ae64e449a1d9a37.ply 683f2b1033b37070a3e6fc2e4e9fddad.ply  ee9ddbd92a65e1a8669cd104872cc35a.ply 6aa835bdf472a5a28dac65057f18f3c9.ply  f58b36cb74d71a2fbda72093f9b5aa73.ply 6bfb7879e048c2286399941a79c1786.ply   f79c474c143cf535c55facedff72a279.ply 6c39f56696a2c9f343739a2b5879ac59.ply  f7b77c7675b54cb27af534cd0698944.ply 6d16ca2914235c18a413908c0e169330.ply  f9718f4cb6d4eac39e42475e9daa66e6.ply 74df7b291536e4b8ddf5f9e6a8568b73.ply  fa3bbc513695d6e1d2c88960a4c1f835.ply 7ca956a41c235a523b12642901083f18.ply  fc9c05f4551c3cd2597505fd7d99b613.ply

Packing Train dinosaur:
1cf1a8987eec28c36130453ea99bdd2e.ply  abeff98fe8b91f65a1243035ea36834a.ply 251cf943220320b58d9e28b82ccb9b64.ply  ae075b3d116942a970a43c2d978e502e.ply 2efaf9918bb612fb2dbc789eef629db.ply   b9bba4d64104ab00943289205ba6a852.ply 3810ba9b6dcffae61981b97966e0ce3e.ply  c37b19f57061b74b8a0623b5a9b46bf5.ply 40a6c699dbdb6eadfcace4d823343363.ply  d52edebda651cb442fc740c916f7e512.ply 40d5e634263946a3b31a52e5df4eca34.ply  dd881dd10ab2e7a823e66a890b580206.ply 441176f9d50ad352f89f49d92c7a9783.ply  f2f3b3aa3c6f5a58c0bf76250610d427.ply 57e9731ff387f2c7fdddc29bb3e72366.ply  f95321b35c2817bdaef005b7f8d10dde.ply 82491f342b2bad4bb7de34dc19372cba.ply  f980ae331c6491c96130453ea99bdd2e.ply 8cd9d7ecc7b9409f79aaa18f9a53207.ply   fb5601b785c655dde623900e9b9f9273.ply 9691f0d4690ec96815433487b54f0d5.ply

Packing Train mug:
128ecbc10df5b05d96eaf1340564a4de.ply  896f1d494bac0ebcdec712af445786fe.ply 141f1db25095b16dcfb3760e4293e310.ply  8b1dca1414ba88cb91986c63a4d7a99a.ply 162201dfe14b73f0281365259d1cf342.ply  9d8c711750a73b06ad1d789f3b2120d0.ply 1c9f9e25c654cbca3c71bf3f4dd78475.ply  a35a92dcc481a994e45e693d882ad8f.ply 22a9f37e6534da451012cc02986a86c3.ply  a8f7a0edd3edc3299e54b4084dc33544.ply 2852b888abae54b0e3523e99fd841f4.ply   b46e89995f4f9cc5161e440f04bd2a2.ply 2997f21fa426e18a6ab1a25d0e8f3590.ply  b6f30c63c946c286cf6897d8875cfd5e.ply 336122c3105440d193e42e2720468bf0.ply  b811555ccf5ef6c4948fa2daa427fe1f.ply 37f56901a07da69dac6b8e58caf61f95.ply  b88bcf33f25c6cb15b4f129f868dedb.ply 387b695db51190d3be276203d0b1a33f.ply  bea77759a3e5f9037ae0031c221d81a4.ply 403fb4eb4fc6235adf0c7dbe7f8f4c8e.ply  bed29baf625ce9145b68309557f3a78c.ply 40f9a6cc6b2c3b3a78060a3a3a55e18f.ply  c453274b341f8c4ec2b9bcaf66ea9919.ply 414772162ef70ec29109ad7f9c200d62.ply  c51b79493419eccdc1584fff35347dc6.ply 44f9c4e1ea3532b8d7b20fded0142d7a.ply  c6bc2c9770a59b5ddd195661813efe58.ply 48e260a614c0fd4434a8988fdcee4fde.ply  c86c80818239a7d8cfdfe3f21f42a111.ply 4f9f31db3c3873692a6f53dd95fd4468.ply  cc5b14ef71e87e9165ba97214ebde03.ply 542235fc88d22e1e3406473757712946.ply  d46b98f63a017578ea456f4bbbc96af9.ply 57f73714cbc425e44ae022a8f6e258a7.ply  d7ba704184d424dfd56d9106430c3fe.ply 59c2a9f6e97ebde9266e6525b6754758.ply  dcec634f18e12427c2c72e575af174cd.ply 5c48d471200d2bf16e8a121e6886e18d.ply  e16a895052da87277f58c33b328479f4.ply 5d803107b8a9aec8724d867867ccd9fb.ply  e8ea2bff9f228aaf4c0fcabe634d03c.ply 5e42608cac0cb5e94962fcaf2d60c3de.ply  e94e46bc5833f2f5e57b873e4f3ef3a4.ply 62634df2ad8f19b87d1b7935311a2ed0.ply  f09e51579600cfbb88b651d2e4ea0846.ply 6aec84952a5ffcf33f60d03e1cb068dc.ply  f1c5b9bb744afd96d6e1954365b10b52.ply 6c379385bf0a23ffdec712af445786fe.ply  f394a2f4d4593b068b44b9dceab64d65.ply 6faf1f04bde838e477f883dde7397db2.ply  f98752f297c37d5a5effb6e5cbfa8e1f.ply 7374ea7fee07f94c86032c4825f3450.ply   fad118b32085f3f2c2c72e575af174cd.ply 73b8b6456221f4ea20d3c05c08e26f.ply feaac65ac950d702f48b9fefd9341bfa.ply

Packing Train wine_glass:
1677ef39627c63dfe99cbea00229006f.ply  86d0f3abbefa91bb8b229d9600c1c2d7.ply 18d97e86cb282236869a35f18d9884c1.ply  8c4da982fb652235687df49cd30b91ce.ply 1adc8b509a8c31242c528d33bca1ac2.ply   91391df0e582d63d202b39ea1c0b23ad.ply 1e196bc2feb78df12babd05184cbfb06.ply  99e9094b908258daf54e1b6f41fdd78a.ply 307bed0948792362b8bfa69ff12f2c13.ply  a2a922f714e6fa4937a692cc86ebcb7c.ply 3626f0f098f5d3b9457ba044c28858b1.ply  a8bfc06799c33472af4429cef16e2ed1.ply 38a1de563bfc282973769674eb74d3d7.ply  ab6c2a310a7d7174764eb68dbf23e35b.ply 3a3b3d325982e37241ce6028a4b3d4cc.ply  ad86cdf0641e395335e1d1ee0e46a9dd.ply 3f6127897c15f4b3d3dfba73f54d35bb.ply  bbfae08c851bd8e431c7c7668b2bd942.ply 3fe63d400a8d7933670c8b003f58e2e6.ply  d17b92ec74bb57e536a5161875b0c6ae.ply 46c8182cb72711e6204e7ce9c2ad318d.ply  d32afe5e3d51e81e52979c729a1f9cc2.ply 566c39df9f86e8b4dcad73f240f03a20.ply  d7a98689aac05176bed06f4e613c0aec.ply 58efcd0392449a67806b90e3f08c9a28.ply  da99c443fddf35e9149c1768202b4e64.ply 5add356c592899bb50fadaaa050ed317.ply  db0d12c1a2c61658362d7fcfd32ff15d.ply 5e109f13e4d34ca2346ba54dc4c0b6f4.ply  dc1e6a14e721817dab8d58596a04230c.ply 74376f4afeb5540daa25bbb65d163ad6.ply  e8c15133bd48a375ce8ed2332763ec5.ply 849ffcf14a4a2496a7f1e0b0ef66fe6f.ply  f42a919e867b86fb9a6e43b878d5b335.ply 86742b0324358074b784faac204319d9.ply  fe3ebaab638f1bc6d043aa8c780141be.ply

Packing Test-1 boat:
3f31208bf7adf03610aa1fb812b74966.ply  e8d1da5077f0fd3754f4583ac182c2.ply 4b1d9f8800d6abc4e39c40af1a23d37f.ply  ead9a5161eb4b6759978215dad30da61.ply 6881c982ed4078baa413908c0e169330.ply  f84196260e558ff5abb59ca95d19016.ply 8b3cf3842a88136c67c16c85d5f8c7d7.ply

Packing Test-1 bottle:
1cf98e5b6fff5471c8724d5673a063a6.ply  8cf1cc180de4ecd0a826d25a0acb0476.ply 1e5abf0465d97d826118a17db9de8c0.ply   91a1f4e7a5eab4eab05e5be75e72ca3c.ply 3f91158956ad7db0322747720d7d37e8.ply  a6f3fe8a16c559ac2ecfb32ca82c5615.ply 47ede0c10e36fe309029fc9335eeb05c.ply  af3dda1cfe61d0fc9403b0d0536a04af.ply 523cddb320608c09a37f3fc191551700.ply  d74bc917899133e080c257afea181fa2.ply 546111e6869f41aca577e3e5353dd356.ply  dc0926ce09d6ce78eb8e919b102c6c08.ply 6ca2149ac6d3699130612f5c0ef21eb8.ply  e6f95cfb9825c0f65070edcf21eb751c.ply 7984d4980d5b07bceba393d429f71de3.ply  f4851a2835228377e101b7546e3ee8a7.ply 799397068de1ae1c4587d6a85176d7a0.ply  milkbottle.ply

Packing Test-1 box:
1b7d79f4ddec642588b0cf16ef014df6.ply  bd4d2a9add3d30e3ef5281b9b6bee86.ply 4af4b9f27740c2183c3fecc5f1e50b1c.ply  defe74c05001775c13d6ba6128d0b598.ply 66b5920f35cc7c733286aedf3f3ac50d.ply  fac53986d7cd442bca972daa503b3095.ply

Packing Test-1 car:
1e4932c0440051a1969cbec8c849925a.ply  9d3a628729606f536be88557189c3c9e.ply 201ec827a6b490641687cc12e43d40c.ply   a67beb5c743b7405c94eafb7767d583b.ply 4b0bce212c843ac9700a0825786d4b5c.ply  b246281328e7c7c691260504695f0c4a.ply 4ff72cafd64c684b434cc282cf621d42.ply  bf9ced34b7175ea2253c03b7df20edd5.ply 51ae6558e6914920253c03b7df20edd5.ply  d48462269ea4c61b6aca84d27b14974e.ply 56671d3bc5fb424ffbb064220a1966c.ply   d6ab9c1b1126cef47c79723ef78c9a81.ply 7958d98f09eae9b2c27928561da17dc0.ply  f14842281e9bf1e2d8f3210e78d03c49.ply 7a228c811e7e18ad18e1879dc4ad8784.ply  f6d7cd8a000997b1aa69dfdc5532bb13.ply 960b561b29a093faa9ae6ab64611062a.ply

packing Test-1 dinosaur:
2b12f8cb34ca9dab3361ea1cff1976a6.ply  9ffd4219ac922eff40287e006d690b09.ply 47460cdfe06af9a091ca3d51b8599f3c.ply  c6403c9c8544d81274bc6e1b100c14d6.ply 876b14976a9adfa5cdea8940fbb67238.ply  ee177d76ccfb18f56130453ea99bdd2e.ply 89d7cdbd6fb34dcf185e23992ec29fa0.ply

Packing Test-1 mug:
187859d3c3a2fd23f54e1b6f41fdd78a.ply  b7e705de46ebdcc14af54ba5738cb1c5.ply 1eaf8db2dd2b710c7d5b1b70ae595e60.ply  b9c5e403c471e86343e2a554546956a2.ply 34ae0b61b0d8aaf2d7b20fded0142d7a.ply  ba10400c108e5c3f54e1b6f41fdd78a.ply 46ed9dad0440c043d33646b0990bb4a.ply   c39fb75015184c2a0c7f097b1a1f7a5.ply 649a51c711dc7f3b32e150233fdd42e9.ply  cf777e14ca2c7a19b4aad3cc5ce7ee8.ply 6a9b31e1298ca1109c515ccf0f61e75f.ply  d38295b8d83e8cdec712af445786fe.ply 71995893d717598c9de7b195ccfa970.ply   d75af64aa166c24eacbe2257d0988c9c.ply 9737c77d3263062b8ca7a0a01bcd55b6.ply  e6dedae946ff5265a95fb60c110b25aa.ply 99eaa69cf6fe8811dec712af445786fe.ply  ec846432f3ebedf0a6f32a8797e3b9e9.ply a637500654ca8d16c97cfc3e8a6b1d16.ply

Packing Test-1 wine_glass:
25b31dd98896c7cc4e0bddb1ed44ff80.ply  828a337655bfa7237a17701d17454d40.ply 2d90fa30a46efcda9a6e43b878d5b335.ply  910f980956810bc7265d1076b4b6c5c.ply 43710274af717071e3a031805ace4a99.ply  aa4c2f7c121e6c65ab5dfd22277ffcae.ply 445f26ebab09006782b108f8f7d0670d.ply  bfd96383a7b684b837a7774639dab343.ply 48c8f3176fbe7b37384368499a680cf1.ply  daa71ddf643a1b731db1eff0d286b8d1.ply 4d4206e74307d1975af597c14b093f6.ply   ec742ff34906e5e890234001fde0a408.ply

Packing Test-2 airplane:
121e9fceb90440efed79d3bd546890bd.ply  909cd7e95efbb08fb7f76a1208b37d78.ply 133937bd45f953748be6919d4632fec1.ply  9cb21d68582e1c4ec1ccec171a275967.ply 13ac14c5106f359e2d4983c39e7d8c2d.ply  a489ea59d8e34bbc9488dcfe4cd752.ply 18666bda4244d22ca7aff2c3136e8e59.ply  a5ca77bfabfad028b56059027b9b30d2.ply 233384d7137f2639a4ff221811523181.ply  a839884fba87173ce25031ee39d82b94.ply 24968851e483feb237678474be485ca.ply   a9024e1edcb0bfc377c1f99ed297e3ca.ply 24bdf389877fb7f21b1f694e36340ebb.ply  ab6177ac32c1405423db3adbb2f1dce.ply 284e6431669d46fd44797ce00623b3fd.ply  ae9b322ca01ab6e89604f248108a555b.ply 297316f1c8a6f59bd007e7980e2b01ba.ply  af79700be6d874e46cc88316c8ef8570.ply 2c97e6b2c92913cac1ccec171a275967.ply  b42620c214ca7cfc44deee06752bbfaa.ply 3b1e49b4b9cdaa9ff739048ffee50e0.ply   b611ea1285980e7c37678474be485ca.ply 3e8fc3155bbc4225c1ccec171a275967.ply  b75e942cb2e673b31402a75556827747.ply 3fbee164467b933e4e493239c0bd39f9.ply  ba37c8ef604b675be1873a3963e0d14.ply 4486a49959b1ff7a4583d51d6ad2ec6d.ply  c16495a75db30903fc66b6d17679fc1.ply 45c963d64b8ef3de37678474be485ca.ply   c21ace7f754376373028a9e038154bbd.ply 4786c261cfff4bc58bd08ee7a9d43de3.ply  c85d03451af27e4a12aa6b5575a3145c.ply 499cd997bbdfff0fde3d252d47f312ec.ply  cbacce3a17e061251ab9df4be75138d0.ply 4a0ec5f510cf976efbd03b0334a9c716.ply  ce0c461237a21cefdb22b838c125a50b.ply 4c9214d70e0a00c6c1ccec171a275967.ply  cf17d0ce34f09f572722fc1bdd7e0e51.ply 4dfe2199fdd333d09837a3d08a6a1966.ply  d3b9114df1d8a3388e415c6cf89025f0.ply 4eec193345af5f6077c1f99ed297e3ca.ply  d3f253eb6c8d004a2187a5d1956c2f76.ply 5abe1e80846dd485c3e7213e9e8493f0.ply  db96e3af2b07248b84d3db4b0fade3b2.ply 5c74962846d6cd33920ed6df8d81211d.ply  dc03e743739c4f87c27f2d9f006d69eb.ply 5d5aefde5935fc9eaa5d0ddd6a2781ea.ply  e1bb92890775e3c251c8fc68d1e46169.ply 5ff1360e1d4137bdb131c2ccc2e4397.ply   e283bda3d44b6a30785fc017b31be9fd.ply 61dc521330d7a3097bdf7c8888a008b8.ply  e8c26282566f7650e1873a3963e0d14.ply 63e9085d3f92fe95554b92378e03dcec.ply  e8c52e4c6ff0a02f5cc77596a44fdf50.ply 6c3593d0cc398e715d058822e4c8a2a8.ply  ed2aaca045fb1714cd4229f38ad0d015.ply 6c6558e0daf4351a4b00d8f4cd3fae63.ply  ed7dff9e1d6b6df2639033370bc1c2f7.ply 6d432caaa8eba4fb44b2fa2cac0778f5.ply  ef39e1ded6bd03413e537308b716efd.ply 709dbf3263e69d3473873da0bf53928f.ply  f214cf57d5d5634960e1e93111ad3e76.ply 70ae0521d5ca428d7b3d3cfb30f2513e.ply  f36ac9cdcf15ac8497492c4542407e32.ply 70bb20cf86fe6afe76b85a01edd4a109.ply  f816c0cb57d7fa33cf5b9c6e0c7ce895.ply 73702765e51de81ff0d567e1db6c6db5.ply  fa4e8f24d7f0ae1cc1ccec171a275967.ply 7bf9fffb3cf71f34b1781b860c977fce.ply  fddcb2b3d45ce98e641c309f1fd7e183.ply 81440fcd51052844af7d907e4e1905dd.ply  fe0e696b4b21ad38e9122b7f9c6599d.ply 845364350601ca0cbdf0b44e2f8f4e6d.ply  metainfo.xml 8d84adeef1d468104c4c63a5e8bd600a.ply

Packing Test-2 bowl:
2fa46a07683334b3486514fb4bfa48b3.ply  aeb7b4bb33fd43a14e23e9314af9ae57.ply 429a622eac559887bbe43d356df0e955.ply  c25fd49b75c12ef86bbb74f0f607cdd.ply 4845731dbf7522b07492cbf7d8bec255.ply  c2882316451828fd7945873d861da519.ply 4be4184845972fba5ea36d5a57a5a3bb.ply  cfac22c8ca3339b83ce5cb00b21d9584.ply 63787d5ff22ef52a2eb33053525e23a0.ply  dbc35fcbbb90b5b4a7eee628cf5fc3f7.ply 68582543c4c6d0bccfdfe3f21f42a111.ply  dd381b3459767f7b18f18cdcd25d1bbb.ply 6930c4d2e7e880b2e20e92c5b8147e4a.ply  e3e57a94be495771f54e1b6f41fdd78a.ply 6a772d12b98ab61dc26651d9d35b77ca.ply  ecb86f63e92e346a25c70fb1df3f879b.ply 77301246b265a4d3a538bf55f6b58cef.ply  f2cb15fb793e7fa244057c222118625.ply
7c43116dbe35797aea5000d9d3be7992.ply  f44301a26ca7f57c70d5704d62bc4186.ply 899af991203577f019790c8746d79a6f.ply  f44387d8cb8d2e4ebaedc225f2279ecf.ply 8bb057d18e2fcc4779368d1198f406e7.ply  f74bba9a22e044dea3769fcd5f96f4.ply 95ac294f47fd7d87e0b49f27ced29e3.ply   fa23aa60ec51c8e4c40fe5637f0a27e1.ply 960c5c5bff2d3a4bbced73c51e99f8b2.ply  metainfo.xml 9a52843cc89cd208362be90aaa182ec6.ply  sqbowl.ply 9e00bef90f59be40e6b692e613c15ae6.ply  SqrBowl.ply

Packing Test-2 stapler:
1a4daa4904bb4a0949684e7f0bb99f9c.ply  94f8710a6b0eb800bb684e59cc21109c.ply 20c5096ea98cc955920de219c00d1c3b.ply  988004d506b3b75e5e40cf2ee4f6ead.ply 277d71a934d2560bd8c1a9f93afff81e.ply  9a92d1c8ff58b7895eec2fc908a2aeaf.ply 2ef8bb988311eca37536da783d466469.ply  a5cafa3a913187dd8f9dd7647048a0c.ply 31138325cfdf65f7dc0a8ae190d19bd8.ply  b3188e51216de8cce2e4961161b75547.ply 376eb047b40ef4f6a480e3d8fdbd4a92.ply  b5fc0f0125873b3d4db629c0c07799a3.ply 396ada0ab0cd4bba5f16c469ffeb982e.ply  bc910044930d827950a69e53f7a35c05.ply 3ea85c1c8977484c8def06fff01a9692.ply  cd982888f54abe47c3459f09a171783f.ply 471b51bcbbce70c22f81cb887dc35578.ply  cf002585ca227c40781bd00f454a6a39.ply 4dbb5a6cd26d13067964ba700cd97f5.ply   d294455c46b7cb6f31fd93e535713515.ply 4e89486413a767511b4363a46debc223.ply  dbd993a1e538ad05a465092f77585928.ply 586f90ea05f62bd36379618f01f885e3.ply  eb6646d3590c467f1cf3fe6240494178.ply 58a427e5201aa43be00ace8e7c1a5eeb.ply  f39912a4f0516fb897371d1e7cc637f3.ply 6aedeceedd68c20326373299cfdd658.ply   fa61da31d03e8f52391ce5f4018991ce.ply 88ac7b2b3050f1f861f7b52424be58ab.ply  metainfo.xml 8d152be34b41785677937146265c551a.ply

## Training

Use train_segmentations.py, train_completions.py, train_grasp_prediction.py, and train_place_prediction.py to train the network models. They should be run once for each task. (The latter two are only needed for the SP method. This step can be skipped if using pretrained network models from Simulation/tensorflow/models.)

## Testing

Set parameters in the place_*_params.py file. For perception ablation studies, use taskCostFactor = 1 and regraspPlanStepCost = 1. For "No Cost", set all cost factors to 0. For "Step Cost", set regraspPlanStepCost = 1. For GQ, set antipodalCostFactor = 1. For CU, set segmUncertCostFactor = 1, compUncertCostFactor = 1, graspUncertCostFactor = 1, and placeUncertCostFactor = 1. For MC and SP methods, use files with the _mc and _sp suffix..

Next, from the Simulation directory, open N terminals and run python/worker_*.py i for i = 0, ..., N-1. These are worker processes for the arrangement and regrasp planners. For packing, N = 8, for canonical, N = 6, for bottles, N = 2, and for blocks, N = 5.

Then, from the Simulation directory, edit python/place_*.py, and set showViewer = True and showSteps = True. This will visualize the simulation, which is important for the first time running it. Then, run python/place_*.py. If everything is satisfactory, set showViewer = False and showSteps = False and run the scenario. A mat file will be produced with the results. See the matlab folder for scripts for processing the results.

## Review

Check python/geom_pick_place/planner_regrasp.py for how the regrasp planner works. (It is pretty well documented.) This class is inherited by SP and MC regrasp planners. Check environment_*.py for how the grasp and placement conditions are evaluated.
