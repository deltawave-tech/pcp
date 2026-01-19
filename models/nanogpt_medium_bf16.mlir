module {
  func.func @main(%arg0: tensor<256x256xbf16>, %arg1: tensor<128x256xbf16>, %arg2: tensor<256xbf16>, %arg3: tensor<256xbf16>, %arg4: tensor<768x256xbf16>, %arg5: tensor<256x256xbf16>, %arg6: tensor<256xbf16>, %arg7: tensor<256xbf16>, %arg8: tensor<1024x256xbf16>, %arg9: tensor<256x1024xbf16>, %arg10: tensor<256xbf16>, %arg11: tensor<256xbf16>, %arg12: tensor<768x256xbf16>, %arg13: tensor<256x256xbf16>, %arg14: tensor<256xbf16>, %arg15: tensor<256xbf16>, %arg16: tensor<1024x256xbf16>, %arg17: tensor<256x1024xbf16>, %arg18: tensor<256xbf16>, %arg19: tensor<256xbf16>, %arg20: tensor<768x256xbf16>, %arg21: tensor<256x256xbf16>, %arg22: tensor<256xbf16>, %arg23: tensor<256xbf16>, %arg24: tensor<1024x256xbf16>, %arg25: tensor<256x1024xbf16>, %arg26: tensor<256xbf16>, %arg27: tensor<256xbf16>, %arg28: tensor<768x256xbf16>, %arg29: tensor<256x256xbf16>, %arg30: tensor<256xbf16>, %arg31: tensor<256xbf16>, %arg32: tensor<1024x256xbf16>, %arg33: tensor<256x1024xbf16>, %arg34: tensor<256xbf16>, %arg35: tensor<256xbf16>, %arg36: tensor<256x256xbf16>, %arg37: tensor<256xbf16>, %arg38: tensor<64x128xi64>, %arg39: tensor<64x128xi64>) -> tensor<f32> {
    %c = stablehlo.constant dense<0> : tensor<8192xi64>
    %c_0 = stablehlo.constant dense<-100> : tensor<8192xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<128x128xi64>
    %c_2 = stablehlo.constant dense<0> : tensor<128xi64>
    %c_3 = stablehlo.constant dense<1> : tensor<128xi64>
    %c_4 = stablehlo.constant dense<128> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %cst_7 = stablehlo.constant dense<-6.553600e+04> : tensor<bf16>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_11 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<64x128x1024xbf16>
    %cst_13 = arith.constant dense<1> : tensor<1xi64>
    %cst_14 = arith.constant dense<256> : tensor<1xi64>
    %cst_15 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_16 = arith.constant dense<0.17677669529663687> : tensor<1xf64>
    %0 = "stablehlo.gather"(%arg0, %arg38) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256>}> : (tensor<256x256xbf16>, tensor<64x128xi64>) -> tensor<64x128x256xbf16>
    %1 = stablehlo.convert %0 : tensor<64x128x256xbf16>
    %2 = stablehlo.convert %c_5 : (tensor<i64>) -> tensor<f64>
    %3 = stablehlo.convert %c_4 : (tensor<i64>) -> tensor<f64>
    %4 = stablehlo.divide %3, %2 : tensor<f64>
    %5 = stablehlo.ceil %4 : tensor<f64>
    %6 = stablehlo.convert %5 : (tensor<f64>) -> tensor<i64>
    %7 = stablehlo.reshape %6 : (tensor<i64>) -> tensor<1xi64>
    %8 = stablehlo.dynamic_iota %7, dim = 0 : (tensor<1xi64>) -> tensor<128xi64>
    %9 = stablehlo.multiply %8, %c_3 : tensor<128xi64>
    %10 = stablehlo.add %9, %c_2 : tensor<128xi64>
    %11 = "stablehlo.gather"(%arg1, %10) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256>}> : (tensor<128x256xbf16>, tensor<128xi64>) -> tensor<128x256xbf16>
    %12 = stablehlo.convert %11 : tensor<128x256xbf16>
    %13 = stablehlo.broadcast_in_dim %12, dims = [1, 2] : (tensor<128x256xbf16>) -> tensor<64x128x256xbf16>
    %14 = stablehlo.add %1, %13 : tensor<64x128x256xbf16>
    %15 = stablehlo.convert %14 : (tensor<64x128x256xbf16>) -> tensor<64x128x256xf32>
    %16 = stablehlo.convert %15 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %17 = stablehlo.reduce(%16 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %18 = stablehlo.reshape %17 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %19 = stablehlo.convert %cst_14 : (tensor<1xi64>) -> tensor<1xf64>
    %20 = stablehlo.reshape %19 : (tensor<1xf64>) -> tensor<f64>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f64>) -> tensor<64x128x1xf64>
    %22 = stablehlo.divide %18, %21 : tensor<64x128x1xf64>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %24 = stablehlo.subtract %16, %23 : tensor<64x128x256xf64>
    %25 = stablehlo.multiply %24, %24 : tensor<64x128x256xf64>
    %26 = stablehlo.reduce(%25 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %27 = stablehlo.reshape %26 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %28 = stablehlo.divide %27, %21 : tensor<64x128x1xf64>
    %29 = stablehlo.convert %28 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %30 = stablehlo.reduce(%15 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %31 = stablehlo.reshape %30 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %32 = stablehlo.convert %cst_14 : (tensor<1xi64>) -> tensor<1xf32>
    %33 = stablehlo.reshape %32 : (tensor<1xf32>) -> tensor<f32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<64x128x1xf32>
    %35 = stablehlo.divide %31, %34 : tensor<64x128x1xf32>
    %36 = stablehlo.convert %cst_15 : (tensor<1xf64>) -> tensor<1xf32>
    %37 = stablehlo.reshape %36 : (tensor<1xf32>) -> tensor<f32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<64x128x1xf32>
    %39 = stablehlo.add %29, %38 : tensor<64x128x1xf32>
    %40 = stablehlo.rsqrt %39 : tensor<64x128x1xf32>
    %41 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %42 = stablehlo.subtract %15, %41 : tensor<64x128x256xf32>
    %43 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %44 = stablehlo.multiply %42, %43 : tensor<64x128x256xf32>
    %45 = stablehlo.convert %arg2 : (tensor<256xbf16>) -> tensor<256xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %47 = stablehlo.multiply %44, %46 : tensor<64x128x256xf32>
    %48 = stablehlo.convert %arg3 : (tensor<256xbf16>) -> tensor<256xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %50 = stablehlo.add %47, %49 : tensor<64x128x256xf32>
    %51 = stablehlo.convert %50 : (tensor<64x128x256xf32>) -> tensor<64x128x256xbf16>
    %52 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<768x256xbf16>) -> tensor<256x768xbf16>
    %53 = stablehlo.reshape %51 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %54 = stablehlo.dot_general %53, %52, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x768xbf16>) -> tensor<8192x768xbf16>
    %55 = stablehlo.reshape %54 : (tensor<8192x768xbf16>) -> tensor<64x128x768xbf16>
    %56 = stablehlo.slice %55 [0:64, 0:128, 0:256] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %57 = stablehlo.slice %55 [0:64, 0:128, 256:512] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %58 = stablehlo.slice %55 [0:64, 0:128, 512:768] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %59 = stablehlo.reshape %57 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %60 = stablehlo.transpose %59, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %61 = stablehlo.reshape %56 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %62 = stablehlo.transpose %61, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %63 = stablehlo.reshape %58 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %64 = stablehlo.transpose %63, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %65 = stablehlo.transpose %60, dims = [0, 1, 3, 2] : (tensor<64x8x128x32xbf16>) -> tensor<64x8x32x128xbf16>
    %66 = stablehlo.reshape %62 : (tensor<64x8x128x32xbf16>) -> tensor<512x128x32xbf16>
    %67 = stablehlo.reshape %65 : (tensor<64x8x32x128xbf16>) -> tensor<512x32x128xbf16>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2] : (tensor<512x32x128xbf16>) -> tensor<512x32x128xbf16>
    %69 = stablehlo.dot_general %66, %68, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x32xbf16>, tensor<512x32x128xbf16>) -> tensor<512x128x128xbf16>
    %70 = stablehlo.reshape %69 : (tensor<512x128x128xbf16>) -> tensor<64x8x128x128xbf16>
    %71 = stablehlo.convert %cst_16 : (tensor<1xf64>) -> tensor<1xbf16>
    %72 = stablehlo.reshape %71 : (tensor<1xbf16>) -> tensor<bf16>
    %73 = stablehlo.broadcast_in_dim %72, dims = [] : (tensor<bf16>) -> tensor<64x8x128x128xbf16>
    %74 = stablehlo.multiply %70, %73 : tensor<64x8x128x128xbf16>
    %75 = stablehlo.iota dim = 1 : tensor<128x128xi64>
    %76 = stablehlo.iota dim = 0 : tensor<128x128xi64>
    %77 = stablehlo.add %76, %c_1 : tensor<128x128xi64>
    %78 = stablehlo.compare  LE, %75, %77,  SIGNED : (tensor<128x128xi64>, tensor<128x128xi64>) -> tensor<128x128xi1>
    %79 = stablehlo.broadcast_in_dim %78, dims = [0, 1] : (tensor<128x128xi1>) -> tensor<128x128xi1>
    %80 = stablehlo.select %79, %cst_8, %cst_10 : tensor<128x128xi1>, tensor<128x128xf32>
    %81 = stablehlo.reshape %80 : (tensor<128x128xf32>) -> tensor<1x1x128x128xf32>
    %82 = stablehlo.convert %c_6 : (tensor<i64>) -> tensor<f32>
    %83 = stablehlo.broadcast_in_dim %82, dims = [] : (tensor<f32>) -> tensor<1x1x128x128xf32>
    %84 = stablehlo.compare  EQ, %81, %83,  FLOAT : (tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xi1>
    %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x128x128xi1>) -> tensor<64x8x128x128xi1>
    %86 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<64x8x128x128xbf16>
    %87 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2, 3] : (tensor<64x8x128x128xbf16>) -> tensor<64x8x128x128xbf16>
    %88 = stablehlo.select %85, %86, %87 : tensor<64x8x128x128xi1>, tensor<64x8x128x128xbf16>
    %89 = stablehlo.convert %88 : (tensor<64x8x128x128xbf16>) -> tensor<64x8x128x128xf32>
    %90 = stablehlo.reduce(%89 init: %cst_11) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %91 = stablehlo.reshape %90 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %93 = stablehlo.subtract %89, %92 : tensor<64x8x128x128xf32>
    %94 = stablehlo.exponential %93 : tensor<64x8x128x128xf32>
    %95 = stablehlo.reduce(%94 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %96 = stablehlo.reshape %95 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %98 = stablehlo.divide %94, %97 : tensor<64x8x128x128xf32>
    %99 = stablehlo.convert %98 : (tensor<64x8x128x128xf32>) -> tensor<64x8x128x128xbf16>
    %100 = stablehlo.reshape %99 : (tensor<64x8x128x128xbf16>) -> tensor<512x128x128xbf16>
    %101 = stablehlo.reshape %64 : (tensor<64x8x128x32xbf16>) -> tensor<512x128x32xbf16>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2] : (tensor<512x128x32xbf16>) -> tensor<512x128x32xbf16>
    %103 = stablehlo.dot_general %100, %102, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x128xbf16>, tensor<512x128x32xbf16>) -> tensor<512x128x32xbf16>
    %104 = stablehlo.reshape %103 : (tensor<512x128x32xbf16>) -> tensor<64x8x128x32xbf16>
    %105 = stablehlo.transpose %104, dims = [0, 2, 1, 3] : (tensor<64x8x128x32xbf16>) -> tensor<64x128x8x32xbf16>
    %106 = stablehlo.reshape %105 : (tensor<64x128x8x32xbf16>) -> tensor<64x128x256xbf16>
    %107 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %108 = stablehlo.reshape %106 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %109 = stablehlo.dot_general %108, %107, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x256xbf16>) -> tensor<8192x256xbf16>
    %110 = stablehlo.reshape %109 : (tensor<8192x256xbf16>) -> tensor<64x128x256xbf16>
    %111 = stablehlo.add %14, %110 : tensor<64x128x256xbf16>
    %112 = stablehlo.convert %111 : (tensor<64x128x256xbf16>) -> tensor<64x128x256xf32>
    %113 = stablehlo.convert %112 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %114 = stablehlo.reduce(%113 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %115 = stablehlo.reshape %114 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %116 = stablehlo.divide %115, %21 : tensor<64x128x1xf64>
    %117 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %118 = stablehlo.subtract %113, %117 : tensor<64x128x256xf64>
    %119 = stablehlo.multiply %118, %118 : tensor<64x128x256xf64>
    %120 = stablehlo.reduce(%119 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %121 = stablehlo.reshape %120 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %122 = stablehlo.divide %121, %21 : tensor<64x128x1xf64>
    %123 = stablehlo.convert %122 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %124 = stablehlo.reduce(%112 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %125 = stablehlo.reshape %124 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %126 = stablehlo.divide %125, %34 : tensor<64x128x1xf32>
    %127 = stablehlo.add %123, %38 : tensor<64x128x1xf32>
    %128 = stablehlo.rsqrt %127 : tensor<64x128x1xf32>
    %129 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %130 = stablehlo.subtract %112, %129 : tensor<64x128x256xf32>
    %131 = stablehlo.broadcast_in_dim %128, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %132 = stablehlo.multiply %130, %131 : tensor<64x128x256xf32>
    %133 = stablehlo.convert %arg6 : (tensor<256xbf16>) -> tensor<256xf32>
    %134 = stablehlo.broadcast_in_dim %133, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %135 = stablehlo.multiply %132, %134 : tensor<64x128x256xf32>
    %136 = stablehlo.convert %arg7 : (tensor<256xbf16>) -> tensor<256xf32>
    %137 = stablehlo.broadcast_in_dim %136, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %138 = stablehlo.add %135, %137 : tensor<64x128x256xf32>
    %139 = stablehlo.convert %138 : (tensor<64x128x256xf32>) -> tensor<64x128x256xbf16>
    %140 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<1024x256xbf16>) -> tensor<256x1024xbf16>
    %141 = stablehlo.reshape %139 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %142 = stablehlo.dot_general %141, %140, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x1024xbf16>) -> tensor<8192x1024xbf16>
    %143 = stablehlo.reshape %142 : (tensor<8192x1024xbf16>) -> tensor<64x128x1024xbf16>
    %144 = stablehlo.maximum %143, %cst_12 : tensor<64x128x1024xbf16>
    %145 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<256x1024xbf16>) -> tensor<1024x256xbf16>
    %146 = stablehlo.reshape %144 : (tensor<64x128x1024xbf16>) -> tensor<8192x1024xbf16>
    %147 = stablehlo.dot_general %146, %145, contracting_dims = [1] x [0] : (tensor<8192x1024xbf16>, tensor<1024x256xbf16>) -> tensor<8192x256xbf16>
    %148 = stablehlo.reshape %147 : (tensor<8192x256xbf16>) -> tensor<64x128x256xbf16>
    %149 = stablehlo.add %111, %148 : tensor<64x128x256xbf16>
    %150 = stablehlo.convert %149 : (tensor<64x128x256xbf16>) -> tensor<64x128x256xf32>
    %151 = stablehlo.convert %150 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %152 = stablehlo.reduce(%151 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %153 = stablehlo.reshape %152 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %154 = stablehlo.divide %153, %21 : tensor<64x128x1xf64>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %156 = stablehlo.subtract %151, %155 : tensor<64x128x256xf64>
    %157 = stablehlo.multiply %156, %156 : tensor<64x128x256xf64>
    %158 = stablehlo.reduce(%157 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %159 = stablehlo.reshape %158 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %160 = stablehlo.divide %159, %21 : tensor<64x128x1xf64>
    %161 = stablehlo.convert %160 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %162 = stablehlo.reduce(%150 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %163 = stablehlo.reshape %162 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %164 = stablehlo.divide %163, %34 : tensor<64x128x1xf32>
    %165 = stablehlo.add %161, %38 : tensor<64x128x1xf32>
    %166 = stablehlo.rsqrt %165 : tensor<64x128x1xf32>
    %167 = stablehlo.broadcast_in_dim %164, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %168 = stablehlo.subtract %150, %167 : tensor<64x128x256xf32>
    %169 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %170 = stablehlo.multiply %168, %169 : tensor<64x128x256xf32>
    %171 = stablehlo.convert %arg10 : (tensor<256xbf16>) -> tensor<256xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %173 = stablehlo.multiply %170, %172 : tensor<64x128x256xf32>
    %174 = stablehlo.convert %arg11 : (tensor<256xbf16>) -> tensor<256xf32>
    %175 = stablehlo.broadcast_in_dim %174, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %176 = stablehlo.add %173, %175 : tensor<64x128x256xf32>
    %177 = stablehlo.convert %176 : (tensor<64x128x256xf32>) -> tensor<64x128x256xbf16>
    %178 = stablehlo.transpose %arg12, dims = [1, 0] : (tensor<768x256xbf16>) -> tensor<256x768xbf16>
    %179 = stablehlo.reshape %177 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %180 = stablehlo.dot_general %179, %178, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x768xbf16>) -> tensor<8192x768xbf16>
    %181 = stablehlo.reshape %180 : (tensor<8192x768xbf16>) -> tensor<64x128x768xbf16>
    %182 = stablehlo.slice %181 [0:64, 0:128, 0:256] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %183 = stablehlo.slice %181 [0:64, 0:128, 256:512] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %184 = stablehlo.slice %181 [0:64, 0:128, 512:768] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %185 = stablehlo.reshape %183 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %186 = stablehlo.transpose %185, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %187 = stablehlo.reshape %182 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %188 = stablehlo.transpose %187, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %189 = stablehlo.reshape %184 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %190 = stablehlo.transpose %189, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %191 = stablehlo.transpose %186, dims = [0, 1, 3, 2] : (tensor<64x8x128x32xbf16>) -> tensor<64x8x32x128xbf16>
    %192 = stablehlo.reshape %188 : (tensor<64x8x128x32xbf16>) -> tensor<512x128x32xbf16>
    %193 = stablehlo.reshape %191 : (tensor<64x8x32x128xbf16>) -> tensor<512x32x128xbf16>
    %194 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2] : (tensor<512x32x128xbf16>) -> tensor<512x32x128xbf16>
    %195 = stablehlo.dot_general %192, %194, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x32xbf16>, tensor<512x32x128xbf16>) -> tensor<512x128x128xbf16>
    %196 = stablehlo.reshape %195 : (tensor<512x128x128xbf16>) -> tensor<64x8x128x128xbf16>
    %197 = stablehlo.multiply %196, %73 : tensor<64x8x128x128xbf16>
    %198 = stablehlo.broadcast_in_dim %197, dims = [0, 1, 2, 3] : (tensor<64x8x128x128xbf16>) -> tensor<64x8x128x128xbf16>
    %199 = stablehlo.select %85, %86, %198 : tensor<64x8x128x128xi1>, tensor<64x8x128x128xbf16>
    %200 = stablehlo.convert %199 : (tensor<64x8x128x128xbf16>) -> tensor<64x8x128x128xf32>
    %201 = stablehlo.reduce(%200 init: %cst_11) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %202 = stablehlo.reshape %201 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %203 = stablehlo.broadcast_in_dim %202, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %204 = stablehlo.subtract %200, %203 : tensor<64x8x128x128xf32>
    %205 = stablehlo.exponential %204 : tensor<64x8x128x128xf32>
    %206 = stablehlo.reduce(%205 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %207 = stablehlo.reshape %206 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %209 = stablehlo.divide %205, %208 : tensor<64x8x128x128xf32>
    %210 = stablehlo.convert %209 : (tensor<64x8x128x128xf32>) -> tensor<64x8x128x128xbf16>
    %211 = stablehlo.reshape %210 : (tensor<64x8x128x128xbf16>) -> tensor<512x128x128xbf16>
    %212 = stablehlo.reshape %190 : (tensor<64x8x128x32xbf16>) -> tensor<512x128x32xbf16>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2] : (tensor<512x128x32xbf16>) -> tensor<512x128x32xbf16>
    %214 = stablehlo.dot_general %211, %213, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x128xbf16>, tensor<512x128x32xbf16>) -> tensor<512x128x32xbf16>
    %215 = stablehlo.reshape %214 : (tensor<512x128x32xbf16>) -> tensor<64x8x128x32xbf16>
    %216 = stablehlo.transpose %215, dims = [0, 2, 1, 3] : (tensor<64x8x128x32xbf16>) -> tensor<64x128x8x32xbf16>
    %217 = stablehlo.reshape %216 : (tensor<64x128x8x32xbf16>) -> tensor<64x128x256xbf16>
    %218 = stablehlo.transpose %arg13, dims = [1, 0] : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %219 = stablehlo.reshape %217 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %220 = stablehlo.dot_general %219, %218, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x256xbf16>) -> tensor<8192x256xbf16>
    %221 = stablehlo.reshape %220 : (tensor<8192x256xbf16>) -> tensor<64x128x256xbf16>
    %222 = stablehlo.add %149, %221 : tensor<64x128x256xbf16>
    %223 = stablehlo.convert %222 : (tensor<64x128x256xbf16>) -> tensor<64x128x256xf32>
    %224 = stablehlo.convert %223 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %225 = stablehlo.reduce(%224 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %226 = stablehlo.reshape %225 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %227 = stablehlo.divide %226, %21 : tensor<64x128x1xf64>
    %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %229 = stablehlo.subtract %224, %228 : tensor<64x128x256xf64>
    %230 = stablehlo.multiply %229, %229 : tensor<64x128x256xf64>
    %231 = stablehlo.reduce(%230 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %232 = stablehlo.reshape %231 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %233 = stablehlo.divide %232, %21 : tensor<64x128x1xf64>
    %234 = stablehlo.convert %233 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %235 = stablehlo.reduce(%223 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %236 = stablehlo.reshape %235 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %237 = stablehlo.divide %236, %34 : tensor<64x128x1xf32>
    %238 = stablehlo.add %234, %38 : tensor<64x128x1xf32>
    %239 = stablehlo.rsqrt %238 : tensor<64x128x1xf32>
    %240 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %241 = stablehlo.subtract %223, %240 : tensor<64x128x256xf32>
    %242 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %243 = stablehlo.multiply %241, %242 : tensor<64x128x256xf32>
    %244 = stablehlo.convert %arg14 : (tensor<256xbf16>) -> tensor<256xf32>
    %245 = stablehlo.broadcast_in_dim %244, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %246 = stablehlo.multiply %243, %245 : tensor<64x128x256xf32>
    %247 = stablehlo.convert %arg15 : (tensor<256xbf16>) -> tensor<256xf32>
    %248 = stablehlo.broadcast_in_dim %247, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %249 = stablehlo.add %246, %248 : tensor<64x128x256xf32>
    %250 = stablehlo.convert %249 : (tensor<64x128x256xf32>) -> tensor<64x128x256xbf16>
    %251 = stablehlo.transpose %arg16, dims = [1, 0] : (tensor<1024x256xbf16>) -> tensor<256x1024xbf16>
    %252 = stablehlo.reshape %250 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %253 = stablehlo.dot_general %252, %251, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x1024xbf16>) -> tensor<8192x1024xbf16>
    %254 = stablehlo.reshape %253 : (tensor<8192x1024xbf16>) -> tensor<64x128x1024xbf16>
    %255 = stablehlo.maximum %254, %cst_12 : tensor<64x128x1024xbf16>
    %256 = stablehlo.transpose %arg17, dims = [1, 0] : (tensor<256x1024xbf16>) -> tensor<1024x256xbf16>
    %257 = stablehlo.reshape %255 : (tensor<64x128x1024xbf16>) -> tensor<8192x1024xbf16>
    %258 = stablehlo.dot_general %257, %256, contracting_dims = [1] x [0] : (tensor<8192x1024xbf16>, tensor<1024x256xbf16>) -> tensor<8192x256xbf16>
    %259 = stablehlo.reshape %258 : (tensor<8192x256xbf16>) -> tensor<64x128x256xbf16>
    %260 = stablehlo.add %222, %259 : tensor<64x128x256xbf16>
    %261 = stablehlo.convert %260 : (tensor<64x128x256xbf16>) -> tensor<64x128x256xf32>
    %262 = stablehlo.convert %261 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %263 = stablehlo.reduce(%262 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %264 = stablehlo.reshape %263 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %265 = stablehlo.divide %264, %21 : tensor<64x128x1xf64>
    %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %267 = stablehlo.subtract %262, %266 : tensor<64x128x256xf64>
    %268 = stablehlo.multiply %267, %267 : tensor<64x128x256xf64>
    %269 = stablehlo.reduce(%268 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %270 = stablehlo.reshape %269 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %271 = stablehlo.divide %270, %21 : tensor<64x128x1xf64>
    %272 = stablehlo.convert %271 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %273 = stablehlo.reduce(%261 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %274 = stablehlo.reshape %273 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %275 = stablehlo.divide %274, %34 : tensor<64x128x1xf32>
    %276 = stablehlo.add %272, %38 : tensor<64x128x1xf32>
    %277 = stablehlo.rsqrt %276 : tensor<64x128x1xf32>
    %278 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %279 = stablehlo.subtract %261, %278 : tensor<64x128x256xf32>
    %280 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %281 = stablehlo.multiply %279, %280 : tensor<64x128x256xf32>
    %282 = stablehlo.convert %arg18 : (tensor<256xbf16>) -> tensor<256xf32>
    %283 = stablehlo.broadcast_in_dim %282, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %284 = stablehlo.multiply %281, %283 : tensor<64x128x256xf32>
    %285 = stablehlo.convert %arg19 : (tensor<256xbf16>) -> tensor<256xf32>
    %286 = stablehlo.broadcast_in_dim %285, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %287 = stablehlo.add %284, %286 : tensor<64x128x256xf32>
    %288 = stablehlo.convert %287 : (tensor<64x128x256xf32>) -> tensor<64x128x256xbf16>
    %289 = stablehlo.transpose %arg20, dims = [1, 0] : (tensor<768x256xbf16>) -> tensor<256x768xbf16>
    %290 = stablehlo.reshape %288 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %291 = stablehlo.dot_general %290, %289, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x768xbf16>) -> tensor<8192x768xbf16>
    %292 = stablehlo.reshape %291 : (tensor<8192x768xbf16>) -> tensor<64x128x768xbf16>
    %293 = stablehlo.slice %292 [0:64, 0:128, 0:256] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %294 = stablehlo.slice %292 [0:64, 0:128, 256:512] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %295 = stablehlo.slice %292 [0:64, 0:128, 512:768] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %296 = stablehlo.reshape %294 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %297 = stablehlo.transpose %296, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %298 = stablehlo.reshape %293 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %300 = stablehlo.reshape %295 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %302 = stablehlo.transpose %297, dims = [0, 1, 3, 2] : (tensor<64x8x128x32xbf16>) -> tensor<64x8x32x128xbf16>
    %303 = stablehlo.reshape %299 : (tensor<64x8x128x32xbf16>) -> tensor<512x128x32xbf16>
    %304 = stablehlo.reshape %302 : (tensor<64x8x32x128xbf16>) -> tensor<512x32x128xbf16>
    %305 = stablehlo.broadcast_in_dim %304, dims = [0, 1, 2] : (tensor<512x32x128xbf16>) -> tensor<512x32x128xbf16>
    %306 = stablehlo.dot_general %303, %305, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x32xbf16>, tensor<512x32x128xbf16>) -> tensor<512x128x128xbf16>
    %307 = stablehlo.reshape %306 : (tensor<512x128x128xbf16>) -> tensor<64x8x128x128xbf16>
    %308 = stablehlo.multiply %307, %73 : tensor<64x8x128x128xbf16>
    %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2, 3] : (tensor<64x8x128x128xbf16>) -> tensor<64x8x128x128xbf16>
    %310 = stablehlo.select %85, %86, %309 : tensor<64x8x128x128xi1>, tensor<64x8x128x128xbf16>
    %311 = stablehlo.convert %310 : (tensor<64x8x128x128xbf16>) -> tensor<64x8x128x128xf32>
    %312 = stablehlo.reduce(%311 init: %cst_11) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %313 = stablehlo.reshape %312 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %315 = stablehlo.subtract %311, %314 : tensor<64x8x128x128xf32>
    %316 = stablehlo.exponential %315 : tensor<64x8x128x128xf32>
    %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %318 = stablehlo.reshape %317 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %319 = stablehlo.broadcast_in_dim %318, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %320 = stablehlo.divide %316, %319 : tensor<64x8x128x128xf32>
    %321 = stablehlo.convert %320 : (tensor<64x8x128x128xf32>) -> tensor<64x8x128x128xbf16>
    %322 = stablehlo.reshape %321 : (tensor<64x8x128x128xbf16>) -> tensor<512x128x128xbf16>
    %323 = stablehlo.reshape %301 : (tensor<64x8x128x32xbf16>) -> tensor<512x128x32xbf16>
    %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1, 2] : (tensor<512x128x32xbf16>) -> tensor<512x128x32xbf16>
    %325 = stablehlo.dot_general %322, %324, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x128xbf16>, tensor<512x128x32xbf16>) -> tensor<512x128x32xbf16>
    %326 = stablehlo.reshape %325 : (tensor<512x128x32xbf16>) -> tensor<64x8x128x32xbf16>
    %327 = stablehlo.transpose %326, dims = [0, 2, 1, 3] : (tensor<64x8x128x32xbf16>) -> tensor<64x128x8x32xbf16>
    %328 = stablehlo.reshape %327 : (tensor<64x128x8x32xbf16>) -> tensor<64x128x256xbf16>
    %329 = stablehlo.transpose %arg21, dims = [1, 0] : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %330 = stablehlo.reshape %328 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %331 = stablehlo.dot_general %330, %329, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x256xbf16>) -> tensor<8192x256xbf16>
    %332 = stablehlo.reshape %331 : (tensor<8192x256xbf16>) -> tensor<64x128x256xbf16>
    %333 = stablehlo.add %260, %332 : tensor<64x128x256xbf16>
    %334 = stablehlo.convert %333 : (tensor<64x128x256xbf16>) -> tensor<64x128x256xf32>
    %335 = stablehlo.convert %334 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %336 = stablehlo.reduce(%335 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %337 = stablehlo.reshape %336 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %338 = stablehlo.divide %337, %21 : tensor<64x128x1xf64>
    %339 = stablehlo.broadcast_in_dim %338, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %340 = stablehlo.subtract %335, %339 : tensor<64x128x256xf64>
    %341 = stablehlo.multiply %340, %340 : tensor<64x128x256xf64>
    %342 = stablehlo.reduce(%341 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %343 = stablehlo.reshape %342 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %344 = stablehlo.divide %343, %21 : tensor<64x128x1xf64>
    %345 = stablehlo.convert %344 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %346 = stablehlo.reduce(%334 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %347 = stablehlo.reshape %346 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %348 = stablehlo.divide %347, %34 : tensor<64x128x1xf32>
    %349 = stablehlo.add %345, %38 : tensor<64x128x1xf32>
    %350 = stablehlo.rsqrt %349 : tensor<64x128x1xf32>
    %351 = stablehlo.broadcast_in_dim %348, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %352 = stablehlo.subtract %334, %351 : tensor<64x128x256xf32>
    %353 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %354 = stablehlo.multiply %352, %353 : tensor<64x128x256xf32>
    %355 = stablehlo.convert %arg22 : (tensor<256xbf16>) -> tensor<256xf32>
    %356 = stablehlo.broadcast_in_dim %355, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %357 = stablehlo.multiply %354, %356 : tensor<64x128x256xf32>
    %358 = stablehlo.convert %arg23 : (tensor<256xbf16>) -> tensor<256xf32>
    %359 = stablehlo.broadcast_in_dim %358, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %360 = stablehlo.add %357, %359 : tensor<64x128x256xf32>
    %361 = stablehlo.convert %360 : (tensor<64x128x256xf32>) -> tensor<64x128x256xbf16>
    %362 = stablehlo.transpose %arg24, dims = [1, 0] : (tensor<1024x256xbf16>) -> tensor<256x1024xbf16>
    %363 = stablehlo.reshape %361 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %364 = stablehlo.dot_general %363, %362, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x1024xbf16>) -> tensor<8192x1024xbf16>
    %365 = stablehlo.reshape %364 : (tensor<8192x1024xbf16>) -> tensor<64x128x1024xbf16>
    %366 = stablehlo.maximum %365, %cst_12 : tensor<64x128x1024xbf16>
    %367 = stablehlo.transpose %arg25, dims = [1, 0] : (tensor<256x1024xbf16>) -> tensor<1024x256xbf16>
    %368 = stablehlo.reshape %366 : (tensor<64x128x1024xbf16>) -> tensor<8192x1024xbf16>
    %369 = stablehlo.dot_general %368, %367, contracting_dims = [1] x [0] : (tensor<8192x1024xbf16>, tensor<1024x256xbf16>) -> tensor<8192x256xbf16>
    %370 = stablehlo.reshape %369 : (tensor<8192x256xbf16>) -> tensor<64x128x256xbf16>
    %371 = stablehlo.add %333, %370 : tensor<64x128x256xbf16>
    %372 = stablehlo.convert %371 : (tensor<64x128x256xbf16>) -> tensor<64x128x256xf32>
    %373 = stablehlo.convert %372 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %374 = stablehlo.reduce(%373 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %375 = stablehlo.reshape %374 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %376 = stablehlo.divide %375, %21 : tensor<64x128x1xf64>
    %377 = stablehlo.broadcast_in_dim %376, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %378 = stablehlo.subtract %373, %377 : tensor<64x128x256xf64>
    %379 = stablehlo.multiply %378, %378 : tensor<64x128x256xf64>
    %380 = stablehlo.reduce(%379 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %381 = stablehlo.reshape %380 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %382 = stablehlo.divide %381, %21 : tensor<64x128x1xf64>
    %383 = stablehlo.convert %382 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %384 = stablehlo.reduce(%372 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %385 = stablehlo.reshape %384 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %386 = stablehlo.divide %385, %34 : tensor<64x128x1xf32>
    %387 = stablehlo.add %383, %38 : tensor<64x128x1xf32>
    %388 = stablehlo.rsqrt %387 : tensor<64x128x1xf32>
    %389 = stablehlo.broadcast_in_dim %386, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %390 = stablehlo.subtract %372, %389 : tensor<64x128x256xf32>
    %391 = stablehlo.broadcast_in_dim %388, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %392 = stablehlo.multiply %390, %391 : tensor<64x128x256xf32>
    %393 = stablehlo.convert %arg26 : (tensor<256xbf16>) -> tensor<256xf32>
    %394 = stablehlo.broadcast_in_dim %393, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %395 = stablehlo.multiply %392, %394 : tensor<64x128x256xf32>
    %396 = stablehlo.convert %arg27 : (tensor<256xbf16>) -> tensor<256xf32>
    %397 = stablehlo.broadcast_in_dim %396, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %398 = stablehlo.add %395, %397 : tensor<64x128x256xf32>
    %399 = stablehlo.convert %398 : (tensor<64x128x256xf32>) -> tensor<64x128x256xbf16>
    %400 = stablehlo.transpose %arg28, dims = [1, 0] : (tensor<768x256xbf16>) -> tensor<256x768xbf16>
    %401 = stablehlo.reshape %399 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %402 = stablehlo.dot_general %401, %400, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x768xbf16>) -> tensor<8192x768xbf16>
    %403 = stablehlo.reshape %402 : (tensor<8192x768xbf16>) -> tensor<64x128x768xbf16>
    %404 = stablehlo.slice %403 [0:64, 0:128, 0:256] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %405 = stablehlo.slice %403 [0:64, 0:128, 256:512] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %406 = stablehlo.slice %403 [0:64, 0:128, 512:768] : (tensor<64x128x768xbf16>) -> tensor<64x128x256xbf16>
    %407 = stablehlo.reshape %405 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %408 = stablehlo.transpose %407, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %409 = stablehlo.reshape %404 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %411 = stablehlo.reshape %406 : (tensor<64x128x256xbf16>) -> tensor<64x128x8x32xbf16>
    %412 = stablehlo.transpose %411, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xbf16>) -> tensor<64x8x128x32xbf16>
    %413 = stablehlo.transpose %408, dims = [0, 1, 3, 2] : (tensor<64x8x128x32xbf16>) -> tensor<64x8x32x128xbf16>
    %414 = stablehlo.reshape %410 : (tensor<64x8x128x32xbf16>) -> tensor<512x128x32xbf16>
    %415 = stablehlo.reshape %413 : (tensor<64x8x32x128xbf16>) -> tensor<512x32x128xbf16>
    %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1, 2] : (tensor<512x32x128xbf16>) -> tensor<512x32x128xbf16>
    %417 = stablehlo.dot_general %414, %416, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x32xbf16>, tensor<512x32x128xbf16>) -> tensor<512x128x128xbf16>
    %418 = stablehlo.reshape %417 : (tensor<512x128x128xbf16>) -> tensor<64x8x128x128xbf16>
    %419 = stablehlo.multiply %418, %73 : tensor<64x8x128x128xbf16>
    %420 = stablehlo.broadcast_in_dim %419, dims = [0, 1, 2, 3] : (tensor<64x8x128x128xbf16>) -> tensor<64x8x128x128xbf16>
    %421 = stablehlo.select %85, %86, %420 : tensor<64x8x128x128xi1>, tensor<64x8x128x128xbf16>
    %422 = stablehlo.convert %421 : (tensor<64x8x128x128xbf16>) -> tensor<64x8x128x128xf32>
    %423 = stablehlo.reduce(%422 init: %cst_11) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %424 = stablehlo.reshape %423 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %426 = stablehlo.subtract %422, %425 : tensor<64x8x128x128xf32>
    %427 = stablehlo.exponential %426 : tensor<64x8x128x128xf32>
    %428 = stablehlo.reduce(%427 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %429 = stablehlo.reshape %428 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %430 = stablehlo.broadcast_in_dim %429, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %431 = stablehlo.divide %427, %430 : tensor<64x8x128x128xf32>
    %432 = stablehlo.convert %431 : (tensor<64x8x128x128xf32>) -> tensor<64x8x128x128xbf16>
    %433 = stablehlo.reshape %432 : (tensor<64x8x128x128xbf16>) -> tensor<512x128x128xbf16>
    %434 = stablehlo.reshape %412 : (tensor<64x8x128x32xbf16>) -> tensor<512x128x32xbf16>
    %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1, 2] : (tensor<512x128x32xbf16>) -> tensor<512x128x32xbf16>
    %436 = stablehlo.dot_general %433, %435, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x128xbf16>, tensor<512x128x32xbf16>) -> tensor<512x128x32xbf16>
    %437 = stablehlo.reshape %436 : (tensor<512x128x32xbf16>) -> tensor<64x8x128x32xbf16>
    %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] : (tensor<64x8x128x32xbf16>) -> tensor<64x128x8x32xbf16>
    %439 = stablehlo.reshape %438 : (tensor<64x128x8x32xbf16>) -> tensor<64x128x256xbf16>
    %440 = stablehlo.transpose %arg29, dims = [1, 0] : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %441 = stablehlo.reshape %439 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %442 = stablehlo.dot_general %441, %440, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x256xbf16>) -> tensor<8192x256xbf16>
    %443 = stablehlo.reshape %442 : (tensor<8192x256xbf16>) -> tensor<64x128x256xbf16>
    %444 = stablehlo.add %371, %443 : tensor<64x128x256xbf16>
    %445 = stablehlo.convert %444 : (tensor<64x128x256xbf16>) -> tensor<64x128x256xf32>
    %446 = stablehlo.convert %445 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %447 = stablehlo.reduce(%446 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %448 = stablehlo.reshape %447 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %449 = stablehlo.divide %448, %21 : tensor<64x128x1xf64>
    %450 = stablehlo.broadcast_in_dim %449, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %451 = stablehlo.subtract %446, %450 : tensor<64x128x256xf64>
    %452 = stablehlo.multiply %451, %451 : tensor<64x128x256xf64>
    %453 = stablehlo.reduce(%452 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %454 = stablehlo.reshape %453 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %455 = stablehlo.divide %454, %21 : tensor<64x128x1xf64>
    %456 = stablehlo.convert %455 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %457 = stablehlo.reduce(%445 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %458 = stablehlo.reshape %457 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %459 = stablehlo.divide %458, %34 : tensor<64x128x1xf32>
    %460 = stablehlo.add %456, %38 : tensor<64x128x1xf32>
    %461 = stablehlo.rsqrt %460 : tensor<64x128x1xf32>
    %462 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %463 = stablehlo.subtract %445, %462 : tensor<64x128x256xf32>
    %464 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %465 = stablehlo.multiply %463, %464 : tensor<64x128x256xf32>
    %466 = stablehlo.convert %arg30 : (tensor<256xbf16>) -> tensor<256xf32>
    %467 = stablehlo.broadcast_in_dim %466, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %468 = stablehlo.multiply %465, %467 : tensor<64x128x256xf32>
    %469 = stablehlo.convert %arg31 : (tensor<256xbf16>) -> tensor<256xf32>
    %470 = stablehlo.broadcast_in_dim %469, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %471 = stablehlo.add %468, %470 : tensor<64x128x256xf32>
    %472 = stablehlo.convert %471 : (tensor<64x128x256xf32>) -> tensor<64x128x256xbf16>
    %473 = stablehlo.transpose %arg32, dims = [1, 0] : (tensor<1024x256xbf16>) -> tensor<256x1024xbf16>
    %474 = stablehlo.reshape %472 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %475 = stablehlo.dot_general %474, %473, contracting_dims = [1] x [0] : (tensor<8192x256xbf16>, tensor<256x1024xbf16>) -> tensor<8192x1024xbf16>
    %476 = stablehlo.reshape %475 : (tensor<8192x1024xbf16>) -> tensor<64x128x1024xbf16>
    %477 = stablehlo.maximum %476, %cst_12 : tensor<64x128x1024xbf16>
    %478 = stablehlo.transpose %arg33, dims = [1, 0] : (tensor<256x1024xbf16>) -> tensor<1024x256xbf16>
    %479 = stablehlo.reshape %477 : (tensor<64x128x1024xbf16>) -> tensor<8192x1024xbf16>
    %480 = stablehlo.dot_general %479, %478, contracting_dims = [1] x [0] : (tensor<8192x1024xbf16>, tensor<1024x256xbf16>) -> tensor<8192x256xbf16>
    %481 = stablehlo.reshape %480 : (tensor<8192x256xbf16>) -> tensor<64x128x256xbf16>
    %482 = stablehlo.add %444, %481 : tensor<64x128x256xbf16>
    %483 = stablehlo.convert %482 : (tensor<64x128x256xbf16>) -> tensor<64x128x256xf32>
    %484 = stablehlo.convert %483 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %485 = stablehlo.reduce(%484 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %486 = stablehlo.reshape %485 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %487 = stablehlo.divide %486, %21 : tensor<64x128x1xf64>
    %488 = stablehlo.broadcast_in_dim %487, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %489 = stablehlo.subtract %484, %488 : tensor<64x128x256xf64>
    %490 = stablehlo.multiply %489, %489 : tensor<64x128x256xf64>
    %491 = stablehlo.reduce(%490 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %492 = stablehlo.reshape %491 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %493 = stablehlo.divide %492, %21 : tensor<64x128x1xf64>
    %494 = stablehlo.convert %493 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %495 = stablehlo.reduce(%483 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %496 = stablehlo.reshape %495 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %497 = stablehlo.divide %496, %34 : tensor<64x128x1xf32>
    %498 = stablehlo.add %494, %38 : tensor<64x128x1xf32>
    %499 = stablehlo.rsqrt %498 : tensor<64x128x1xf32>
    %500 = stablehlo.broadcast_in_dim %497, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %501 = stablehlo.subtract %483, %500 : tensor<64x128x256xf32>
    %502 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %503 = stablehlo.multiply %501, %502 : tensor<64x128x256xf32>
    %504 = stablehlo.convert %arg34 : (tensor<256xbf16>) -> tensor<256xf32>
    %505 = stablehlo.broadcast_in_dim %504, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %506 = stablehlo.multiply %503, %505 : tensor<64x128x256xf32>
    %507 = stablehlo.convert %arg35 : (tensor<256xbf16>) -> tensor<256xf32>
    %508 = stablehlo.broadcast_in_dim %507, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %509 = stablehlo.add %506, %508 : tensor<64x128x256xf32>
    %510 = stablehlo.convert %509 : (tensor<64x128x256xf32>) -> tensor<64x128x256xbf16>
    %511 = stablehlo.reshape %510 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %512 = stablehlo.transpose %arg36, dims = [1, 0] : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
    %513 = stablehlo.convert %arg37 : (tensor<256xbf16>) -> tensor<256xf32>
    %514 = stablehlo.convert %511 : (tensor<8192x256xbf16>) -> tensor<8192x256xf32>
    %515 = stablehlo.convert %512 : (tensor<256x256xbf16>) -> tensor<256x256xf32>
    %516 = stablehlo.dot_general %514, %515, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x256xf32>) -> tensor<8192x256xf32>
    %517 = stablehlo.convert %cst_13 : (tensor<1xi64>) -> tensor<1xf32>
    %518 = stablehlo.reshape %517 : (tensor<1xf32>) -> tensor<f32>
    %519 = stablehlo.broadcast_in_dim %518, dims = [] : (tensor<f32>) -> tensor<8192x256xf32>
    %520 = stablehlo.multiply %516, %519 : tensor<8192x256xf32>
    %521 = stablehlo.broadcast_in_dim %518, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %522 = stablehlo.multiply %513, %521 : tensor<256xf32>
    %523 = stablehlo.broadcast_in_dim %522, dims = [1] : (tensor<256xf32>) -> tensor<8192x256xf32>
    %524 = stablehlo.add %520, %523 : tensor<8192x256xf32>
    %525 = stablehlo.convert %524 : (tensor<8192x256xf32>) -> tensor<8192x256xbf16>
    %526 = stablehlo.reshape %525 : (tensor<8192x256xbf16>) -> tensor<64x128x256xbf16>
    %527 = stablehlo.reshape %526 : (tensor<64x128x256xbf16>) -> tensor<8192x256xbf16>
    %528 = stablehlo.convert %527 : (tensor<8192x256xbf16>) -> tensor<8192x256xf32>
    %529 = stablehlo.reshape %arg39 : (tensor<64x128xi64>) -> tensor<8192xi64>
    %530 = stablehlo.reduce(%528 init: %cst_11) applies stablehlo.maximum across dimensions = [1] : (tensor<8192x256xf32>, tensor<f32>) -> tensor<8192xf32>
    %531 = stablehlo.reshape %530 : (tensor<8192xf32>) -> tensor<8192x1xf32>
    %532 = stablehlo.broadcast_in_dim %531, dims = [0, 1] : (tensor<8192x1xf32>) -> tensor<8192x256xf32>
    %533 = stablehlo.subtract %528, %532 : tensor<8192x256xf32>
    %534 = stablehlo.exponential %533 : tensor<8192x256xf32>
    %535 = stablehlo.reduce(%534 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8192x256xf32>, tensor<f32>) -> tensor<8192xf32>
    %536 = stablehlo.reshape %535 : (tensor<8192xf32>) -> tensor<8192x1xf32>
    %537 = stablehlo.log %536 : tensor<8192x1xf32>
    %538 = stablehlo.broadcast_in_dim %537, dims = [0, 1] : (tensor<8192x1xf32>) -> tensor<8192x256xf32>
    %539 = stablehlo.subtract %533, %538 : tensor<8192x256xf32>
    %540 = stablehlo.compare  NE, %529, %c_0,  SIGNED : (tensor<8192xi64>, tensor<8192xi64>) -> tensor<8192xi1>
    %541 = stablehlo.broadcast_in_dim %540, dims = [0] : (tensor<8192xi1>) -> tensor<8192xi1>
    %542 = stablehlo.broadcast_in_dim %529, dims = [0] : (tensor<8192xi64>) -> tensor<8192xi64>
    %543 = stablehlo.select %541, %542, %c : tensor<8192xi1>, tensor<8192xi64>
    %544 = stablehlo.reshape %543 : (tensor<8192xi64>) -> tensor<8192x1xi64>
    %545 = stablehlo.iota dim = 0 : tensor<8192x1x1xi64>
    %546 = stablehlo.reshape %544 : (tensor<8192x1xi64>) -> tensor<8192x1x1xi64>
    %547 = stablehlo.concatenate %545, %546, dim = 2 : (tensor<8192x1x1xi64>, tensor<8192x1x1xi64>) -> tensor<8192x1x2xi64>
    %548 = "stablehlo.gather"(%539, %547) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<8192x256xf32>, tensor<8192x1x2xi64>) -> tensor<8192x1xf32>
    %549 = stablehlo.reshape %548 : (tensor<8192x1xf32>) -> tensor<8192xf32>
    %550 = stablehlo.negate %549 : tensor<8192xf32>
    %551 = stablehlo.broadcast_in_dim %550, dims = [0] : (tensor<8192xf32>) -> tensor<8192xf32>
    %552 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8192xf32>
    %553 = stablehlo.select %541, %551, %552 : tensor<8192xi1>, tensor<8192xf32>
    %554 = stablehlo.convert %540 : (tensor<8192xi1>) -> tensor<8192xi64>
    %555 = stablehlo.reduce(%554 init: %c_6) applies stablehlo.add across dimensions = [0] : (tensor<8192xi64>, tensor<i64>) -> tensor<i64>
    %556 = stablehlo.convert %555 : (tensor<i64>) -> tensor<f32>
    %557 = stablehlo.reduce(%553 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<8192xf32>, tensor<f32>) -> tensor<f32>
    %558 = stablehlo.divide %557, %556 : tensor<f32>
    return %558 : tensor<f32>
  }
}
