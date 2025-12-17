module {
  func.func @main(%arg0: tensor<256x256xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<768x256xf32>, %arg5: tensor<256x256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>, %arg8: tensor<1024x256xf32>, %arg9: tensor<256x1024xf32>, %arg10: tensor<256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<768x256xf32>, %arg13: tensor<256x256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256xf32>, %arg16: tensor<1024x256xf32>, %arg17: tensor<256x1024xf32>, %arg18: tensor<256xf32>, %arg19: tensor<256xf32>, %arg20: tensor<768x256xf32>, %arg21: tensor<256x256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<1024x256xf32>, %arg25: tensor<256x1024xf32>, %arg26: tensor<256xf32>, %arg27: tensor<256xf32>, %arg28: tensor<768x256xf32>, %arg29: tensor<256x256xf32>, %arg30: tensor<256xf32>, %arg31: tensor<256xf32>, %arg32: tensor<1024x256xf32>, %arg33: tensor<256x1024xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x256xf32>, %arg37: tensor<256xf32>, %arg38: tensor<64x128xi64>, %arg39: tensor<64x128xi64>) -> tensor<f32> {
    %c = stablehlo.constant dense<0> : tensor<8192xi64>
    %c_0 = stablehlo.constant dense<-100> : tensor<8192xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<128x128xi64>
    %c_2 = stablehlo.constant dense<0> : tensor<128xi64>
    %c_3 = stablehlo.constant dense<1> : tensor<128xi64>
    %c_4 = stablehlo.constant dense<128> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %cst_7 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<64x128x1024xf32>
    %cst_12 = arith.constant dense<1> : tensor<1xi64>
    %cst_13 = arith.constant dense<256> : tensor<1xi64>
    %cst_14 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_15 = arith.constant dense<0.17677669529663687> : tensor<1xf64>
    %0 = "stablehlo.gather"(%arg0, %arg38) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256>}> : (tensor<256x256xf32>, tensor<64x128xi64>) -> tensor<64x128x256xf32>
    %1 = stablehlo.convert %0 : tensor<64x128x256xf32>
    %2 = stablehlo.convert %c_5 : (tensor<i64>) -> tensor<f64>
    %3 = stablehlo.convert %c_4 : (tensor<i64>) -> tensor<f64>
    %4 = stablehlo.divide %3, %2 : tensor<f64>
    %5 = stablehlo.ceil %4 : tensor<f64>
    %6 = stablehlo.convert %5 : (tensor<f64>) -> tensor<i64>
    %7 = stablehlo.reshape %6 : (tensor<i64>) -> tensor<1xi64>
    %8 = stablehlo.dynamic_iota %7, dim = 0 : (tensor<1xi64>) -> tensor<128xi64>
    %9 = stablehlo.multiply %8, %c_3 : tensor<128xi64>
    %10 = stablehlo.add %9, %c_2 : tensor<128xi64>
    %11 = "stablehlo.gather"(%arg1, %10) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256>}> : (tensor<128x256xf32>, tensor<128xi64>) -> tensor<128x256xf32>
    %12 = stablehlo.convert %11 : tensor<128x256xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [1, 2] : (tensor<128x256xf32>) -> tensor<64x128x256xf32>
    %14 = stablehlo.add %1, %13 : tensor<64x128x256xf32>
    %15 = stablehlo.convert %14 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %16 = stablehlo.reduce(%15 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %17 = stablehlo.reshape %16 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %18 = stablehlo.convert %cst_13 : (tensor<1xi64>) -> tensor<1xf64>
    %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f64>) -> tensor<64x128x1xf64>
    %21 = stablehlo.divide %17, %20 : tensor<64x128x1xf64>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %23 = stablehlo.subtract %15, %22 : tensor<64x128x256xf64>
    %24 = stablehlo.multiply %23, %23 : tensor<64x128x256xf64>
    %25 = stablehlo.reduce(%24 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %26 = stablehlo.reshape %25 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %27 = stablehlo.divide %26, %20 : tensor<64x128x1xf64>
    %28 = stablehlo.convert %27 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %29 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %30 = stablehlo.reshape %29 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %31 = stablehlo.convert %cst_13 : (tensor<1xi64>) -> tensor<1xf32>
    %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<64x128x1xf32>
    %34 = stablehlo.divide %30, %33 : tensor<64x128x1xf32>
    %35 = stablehlo.convert %cst_14 : (tensor<1xf64>) -> tensor<1xf32>
    %36 = stablehlo.reshape %35 : (tensor<1xf32>) -> tensor<f32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<f32>) -> tensor<64x128x1xf32>
    %38 = stablehlo.add %28, %37 : tensor<64x128x1xf32>
    %39 = stablehlo.rsqrt %38 : tensor<64x128x1xf32>
    %40 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %41 = stablehlo.subtract %14, %40 : tensor<64x128x256xf32>
    %42 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<64x128x256xf32>
    %44 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<64x128x256xf32>
    %46 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %47 = stablehlo.add %45, %46 : tensor<64x128x256xf32>
    %48 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %49 = stablehlo.reshape %47 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %50 = stablehlo.dot_general %49, %48, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x768xf32>) -> tensor<8192x768xf32>
    %51 = stablehlo.reshape %50 : (tensor<8192x768xf32>) -> tensor<64x128x768xf32>
    %52 = stablehlo.slice %51 [0:64, 0:128, 0:256] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %53 = stablehlo.slice %51 [0:64, 0:128, 256:512] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %54 = stablehlo.slice %51 [0:64, 0:128, 512:768] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %55 = stablehlo.reshape %53 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %56 = stablehlo.transpose %55, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %57 = stablehlo.reshape %52 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %58 = stablehlo.transpose %57, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %59 = stablehlo.reshape %54 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %60 = stablehlo.transpose %59, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %61 = stablehlo.transpose %56, dims = [0, 1, 3, 2] : (tensor<64x8x128x32xf32>) -> tensor<64x8x32x128xf32>
    %62 = stablehlo.reshape %58 : (tensor<64x8x128x32xf32>) -> tensor<512x128x32xf32>
    %63 = stablehlo.reshape %61 : (tensor<64x8x32x128xf32>) -> tensor<512x32x128xf32>
    %64 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2] : (tensor<512x32x128xf32>) -> tensor<512x32x128xf32>
    %65 = stablehlo.dot_general %62, %64, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x32xf32>, tensor<512x32x128xf32>) -> tensor<512x128x128xf32>
    %66 = stablehlo.reshape %65 : (tensor<512x128x128xf32>) -> tensor<64x8x128x128xf32>
    %67 = stablehlo.convert %cst_15 : (tensor<1xf64>) -> tensor<1xf32>
    %68 = stablehlo.reshape %67 : (tensor<1xf32>) -> tensor<f32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f32>) -> tensor<64x8x128x128xf32>
    %70 = stablehlo.multiply %66, %69 : tensor<64x8x128x128xf32>
    %71 = stablehlo.iota dim = 1 : tensor<128x128xi64>
    %72 = stablehlo.iota dim = 0 : tensor<128x128xi64>
    %73 = stablehlo.add %72, %c_1 : tensor<128x128xi64>
    %74 = stablehlo.compare  LE, %71, %73,  SIGNED : (tensor<128x128xi64>, tensor<128x128xi64>) -> tensor<128x128xi1>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1] : (tensor<128x128xi1>) -> tensor<128x128xi1>
    %76 = stablehlo.select %75, %cst_8, %cst_10 : tensor<128x128xi1>, tensor<128x128xf32>
    %77 = stablehlo.reshape %76 : (tensor<128x128xf32>) -> tensor<1x1x128x128xf32>
    %78 = stablehlo.convert %c_6 : (tensor<i64>) -> tensor<f32>
    %79 = stablehlo.broadcast_in_dim %78, dims = [] : (tensor<f32>) -> tensor<1x1x128x128xf32>
    %80 = stablehlo.compare  EQ, %77, %79,  FLOAT : (tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xi1>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2, 3] : (tensor<1x1x128x128xi1>) -> tensor<64x8x128x128xi1>
    %82 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x8x128x128xf32>
    %83 = stablehlo.broadcast_in_dim %70, dims = [0, 1, 2, 3] : (tensor<64x8x128x128xf32>) -> tensor<64x8x128x128xf32>
    %84 = stablehlo.select %81, %82, %83 : tensor<64x8x128x128xi1>, tensor<64x8x128x128xf32>
    %85 = stablehlo.reduce(%84 init: %cst_7) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %86 = stablehlo.reshape %85 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %88 = stablehlo.subtract %84, %87 : tensor<64x8x128x128xf32>
    %89 = stablehlo.exponential %88 : tensor<64x8x128x128xf32>
    %90 = stablehlo.reduce(%89 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %91 = stablehlo.reshape %90 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %93 = stablehlo.divide %89, %92 : tensor<64x8x128x128xf32>
    %94 = stablehlo.reshape %93 : (tensor<64x8x128x128xf32>) -> tensor<512x128x128xf32>
    %95 = stablehlo.reshape %60 : (tensor<64x8x128x32xf32>) -> tensor<512x128x32xf32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2] : (tensor<512x128x32xf32>) -> tensor<512x128x32xf32>
    %97 = stablehlo.dot_general %94, %96, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x128xf32>, tensor<512x128x32xf32>) -> tensor<512x128x32xf32>
    %98 = stablehlo.reshape %97 : (tensor<512x128x32xf32>) -> tensor<64x8x128x32xf32>
    %99 = stablehlo.transpose %98, dims = [0, 2, 1, 3] : (tensor<64x8x128x32xf32>) -> tensor<64x128x8x32xf32>
    %100 = stablehlo.reshape %99 : (tensor<64x128x8x32xf32>) -> tensor<64x128x256xf32>
    %101 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %102 = stablehlo.reshape %100 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %103 = stablehlo.dot_general %102, %101, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x256xf32>) -> tensor<8192x256xf32>
    %104 = stablehlo.reshape %103 : (tensor<8192x256xf32>) -> tensor<64x128x256xf32>
    %105 = stablehlo.add %14, %104 : tensor<64x128x256xf32>
    %106 = stablehlo.convert %105 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %107 = stablehlo.reduce(%106 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %108 = stablehlo.reshape %107 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %109 = stablehlo.divide %108, %20 : tensor<64x128x1xf64>
    %110 = stablehlo.broadcast_in_dim %109, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %111 = stablehlo.subtract %106, %110 : tensor<64x128x256xf64>
    %112 = stablehlo.multiply %111, %111 : tensor<64x128x256xf64>
    %113 = stablehlo.reduce(%112 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %114 = stablehlo.reshape %113 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %115 = stablehlo.divide %114, %20 : tensor<64x128x1xf64>
    %116 = stablehlo.convert %115 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %117 = stablehlo.reduce(%105 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %118 = stablehlo.reshape %117 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %119 = stablehlo.divide %118, %33 : tensor<64x128x1xf32>
    %120 = stablehlo.add %116, %37 : tensor<64x128x1xf32>
    %121 = stablehlo.rsqrt %120 : tensor<64x128x1xf32>
    %122 = stablehlo.broadcast_in_dim %119, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %123 = stablehlo.subtract %105, %122 : tensor<64x128x256xf32>
    %124 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %125 = stablehlo.multiply %123, %124 : tensor<64x128x256xf32>
    %126 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %127 = stablehlo.multiply %125, %126 : tensor<64x128x256xf32>
    %128 = stablehlo.broadcast_in_dim %arg7, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %129 = stablehlo.add %127, %128 : tensor<64x128x256xf32>
    %130 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %131 = stablehlo.reshape %129 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %132 = stablehlo.dot_general %131, %130, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x1024xf32>) -> tensor<8192x1024xf32>
    %133 = stablehlo.reshape %132 : (tensor<8192x1024xf32>) -> tensor<64x128x1024xf32>
    %134 = stablehlo.maximum %133, %cst_11 : tensor<64x128x1024xf32>
    %135 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
    %136 = stablehlo.reshape %134 : (tensor<64x128x1024xf32>) -> tensor<8192x1024xf32>
    %137 = stablehlo.dot_general %136, %135, contracting_dims = [1] x [0] : (tensor<8192x1024xf32>, tensor<1024x256xf32>) -> tensor<8192x256xf32>
    %138 = stablehlo.reshape %137 : (tensor<8192x256xf32>) -> tensor<64x128x256xf32>
    %139 = stablehlo.add %105, %138 : tensor<64x128x256xf32>
    %140 = stablehlo.convert %139 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %141 = stablehlo.reduce(%140 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %142 = stablehlo.reshape %141 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %143 = stablehlo.divide %142, %20 : tensor<64x128x1xf64>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %145 = stablehlo.subtract %140, %144 : tensor<64x128x256xf64>
    %146 = stablehlo.multiply %145, %145 : tensor<64x128x256xf64>
    %147 = stablehlo.reduce(%146 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %148 = stablehlo.reshape %147 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %149 = stablehlo.divide %148, %20 : tensor<64x128x1xf64>
    %150 = stablehlo.convert %149 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %151 = stablehlo.reduce(%139 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %152 = stablehlo.reshape %151 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %153 = stablehlo.divide %152, %33 : tensor<64x128x1xf32>
    %154 = stablehlo.add %150, %37 : tensor<64x128x1xf32>
    %155 = stablehlo.rsqrt %154 : tensor<64x128x1xf32>
    %156 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %157 = stablehlo.subtract %139, %156 : tensor<64x128x256xf32>
    %158 = stablehlo.broadcast_in_dim %155, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %159 = stablehlo.multiply %157, %158 : tensor<64x128x256xf32>
    %160 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %161 = stablehlo.multiply %159, %160 : tensor<64x128x256xf32>
    %162 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %163 = stablehlo.add %161, %162 : tensor<64x128x256xf32>
    %164 = stablehlo.transpose %arg12, dims = [1, 0] : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %165 = stablehlo.reshape %163 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %166 = stablehlo.dot_general %165, %164, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x768xf32>) -> tensor<8192x768xf32>
    %167 = stablehlo.reshape %166 : (tensor<8192x768xf32>) -> tensor<64x128x768xf32>
    %168 = stablehlo.slice %167 [0:64, 0:128, 0:256] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %169 = stablehlo.slice %167 [0:64, 0:128, 256:512] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %170 = stablehlo.slice %167 [0:64, 0:128, 512:768] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %171 = stablehlo.reshape %169 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %172 = stablehlo.transpose %171, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %173 = stablehlo.reshape %168 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %174 = stablehlo.transpose %173, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %175 = stablehlo.reshape %170 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %176 = stablehlo.transpose %175, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %177 = stablehlo.transpose %172, dims = [0, 1, 3, 2] : (tensor<64x8x128x32xf32>) -> tensor<64x8x32x128xf32>
    %178 = stablehlo.reshape %174 : (tensor<64x8x128x32xf32>) -> tensor<512x128x32xf32>
    %179 = stablehlo.reshape %177 : (tensor<64x8x32x128xf32>) -> tensor<512x32x128xf32>
    %180 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2] : (tensor<512x32x128xf32>) -> tensor<512x32x128xf32>
    %181 = stablehlo.dot_general %178, %180, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x32xf32>, tensor<512x32x128xf32>) -> tensor<512x128x128xf32>
    %182 = stablehlo.reshape %181 : (tensor<512x128x128xf32>) -> tensor<64x8x128x128xf32>
    %183 = stablehlo.multiply %182, %69 : tensor<64x8x128x128xf32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0, 1, 2, 3] : (tensor<64x8x128x128xf32>) -> tensor<64x8x128x128xf32>
    %185 = stablehlo.select %81, %82, %184 : tensor<64x8x128x128xi1>, tensor<64x8x128x128xf32>
    %186 = stablehlo.reduce(%185 init: %cst_7) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %187 = stablehlo.reshape %186 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %189 = stablehlo.subtract %185, %188 : tensor<64x8x128x128xf32>
    %190 = stablehlo.exponential %189 : tensor<64x8x128x128xf32>
    %191 = stablehlo.reduce(%190 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %192 = stablehlo.reshape %191 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %194 = stablehlo.divide %190, %193 : tensor<64x8x128x128xf32>
    %195 = stablehlo.reshape %194 : (tensor<64x8x128x128xf32>) -> tensor<512x128x128xf32>
    %196 = stablehlo.reshape %176 : (tensor<64x8x128x32xf32>) -> tensor<512x128x32xf32>
    %197 = stablehlo.broadcast_in_dim %196, dims = [0, 1, 2] : (tensor<512x128x32xf32>) -> tensor<512x128x32xf32>
    %198 = stablehlo.dot_general %195, %197, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x128xf32>, tensor<512x128x32xf32>) -> tensor<512x128x32xf32>
    %199 = stablehlo.reshape %198 : (tensor<512x128x32xf32>) -> tensor<64x8x128x32xf32>
    %200 = stablehlo.transpose %199, dims = [0, 2, 1, 3] : (tensor<64x8x128x32xf32>) -> tensor<64x128x8x32xf32>
    %201 = stablehlo.reshape %200 : (tensor<64x128x8x32xf32>) -> tensor<64x128x256xf32>
    %202 = stablehlo.transpose %arg13, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %203 = stablehlo.reshape %201 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %204 = stablehlo.dot_general %203, %202, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x256xf32>) -> tensor<8192x256xf32>
    %205 = stablehlo.reshape %204 : (tensor<8192x256xf32>) -> tensor<64x128x256xf32>
    %206 = stablehlo.add %139, %205 : tensor<64x128x256xf32>
    %207 = stablehlo.convert %206 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %208 = stablehlo.reduce(%207 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %209 = stablehlo.reshape %208 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %210 = stablehlo.divide %209, %20 : tensor<64x128x1xf64>
    %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %212 = stablehlo.subtract %207, %211 : tensor<64x128x256xf64>
    %213 = stablehlo.multiply %212, %212 : tensor<64x128x256xf64>
    %214 = stablehlo.reduce(%213 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %215 = stablehlo.reshape %214 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %216 = stablehlo.divide %215, %20 : tensor<64x128x1xf64>
    %217 = stablehlo.convert %216 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %218 = stablehlo.reduce(%206 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %219 = stablehlo.reshape %218 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %220 = stablehlo.divide %219, %33 : tensor<64x128x1xf32>
    %221 = stablehlo.add %217, %37 : tensor<64x128x1xf32>
    %222 = stablehlo.rsqrt %221 : tensor<64x128x1xf32>
    %223 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %224 = stablehlo.subtract %206, %223 : tensor<64x128x256xf32>
    %225 = stablehlo.broadcast_in_dim %222, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %226 = stablehlo.multiply %224, %225 : tensor<64x128x256xf32>
    %227 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %228 = stablehlo.multiply %226, %227 : tensor<64x128x256xf32>
    %229 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %230 = stablehlo.add %228, %229 : tensor<64x128x256xf32>
    %231 = stablehlo.transpose %arg16, dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %232 = stablehlo.reshape %230 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %233 = stablehlo.dot_general %232, %231, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x1024xf32>) -> tensor<8192x1024xf32>
    %234 = stablehlo.reshape %233 : (tensor<8192x1024xf32>) -> tensor<64x128x1024xf32>
    %235 = stablehlo.maximum %234, %cst_11 : tensor<64x128x1024xf32>
    %236 = stablehlo.transpose %arg17, dims = [1, 0] : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
    %237 = stablehlo.reshape %235 : (tensor<64x128x1024xf32>) -> tensor<8192x1024xf32>
    %238 = stablehlo.dot_general %237, %236, contracting_dims = [1] x [0] : (tensor<8192x1024xf32>, tensor<1024x256xf32>) -> tensor<8192x256xf32>
    %239 = stablehlo.reshape %238 : (tensor<8192x256xf32>) -> tensor<64x128x256xf32>
    %240 = stablehlo.add %206, %239 : tensor<64x128x256xf32>
    %241 = stablehlo.convert %240 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %242 = stablehlo.reduce(%241 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %243 = stablehlo.reshape %242 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %244 = stablehlo.divide %243, %20 : tensor<64x128x1xf64>
    %245 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %246 = stablehlo.subtract %241, %245 : tensor<64x128x256xf64>
    %247 = stablehlo.multiply %246, %246 : tensor<64x128x256xf64>
    %248 = stablehlo.reduce(%247 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %249 = stablehlo.reshape %248 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %250 = stablehlo.divide %249, %20 : tensor<64x128x1xf64>
    %251 = stablehlo.convert %250 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %252 = stablehlo.reduce(%240 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %253 = stablehlo.reshape %252 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %254 = stablehlo.divide %253, %33 : tensor<64x128x1xf32>
    %255 = stablehlo.add %251, %37 : tensor<64x128x1xf32>
    %256 = stablehlo.rsqrt %255 : tensor<64x128x1xf32>
    %257 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %258 = stablehlo.subtract %240, %257 : tensor<64x128x256xf32>
    %259 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %260 = stablehlo.multiply %258, %259 : tensor<64x128x256xf32>
    %261 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %262 = stablehlo.multiply %260, %261 : tensor<64x128x256xf32>
    %263 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %264 = stablehlo.add %262, %263 : tensor<64x128x256xf32>
    %265 = stablehlo.transpose %arg20, dims = [1, 0] : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %266 = stablehlo.reshape %264 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %267 = stablehlo.dot_general %266, %265, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x768xf32>) -> tensor<8192x768xf32>
    %268 = stablehlo.reshape %267 : (tensor<8192x768xf32>) -> tensor<64x128x768xf32>
    %269 = stablehlo.slice %268 [0:64, 0:128, 0:256] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %270 = stablehlo.slice %268 [0:64, 0:128, 256:512] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %271 = stablehlo.slice %268 [0:64, 0:128, 512:768] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %272 = stablehlo.reshape %270 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %273 = stablehlo.transpose %272, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %274 = stablehlo.reshape %269 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %275 = stablehlo.transpose %274, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %276 = stablehlo.reshape %271 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %277 = stablehlo.transpose %276, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %278 = stablehlo.transpose %273, dims = [0, 1, 3, 2] : (tensor<64x8x128x32xf32>) -> tensor<64x8x32x128xf32>
    %279 = stablehlo.reshape %275 : (tensor<64x8x128x32xf32>) -> tensor<512x128x32xf32>
    %280 = stablehlo.reshape %278 : (tensor<64x8x32x128xf32>) -> tensor<512x32x128xf32>
    %281 = stablehlo.broadcast_in_dim %280, dims = [0, 1, 2] : (tensor<512x32x128xf32>) -> tensor<512x32x128xf32>
    %282 = stablehlo.dot_general %279, %281, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x32xf32>, tensor<512x32x128xf32>) -> tensor<512x128x128xf32>
    %283 = stablehlo.reshape %282 : (tensor<512x128x128xf32>) -> tensor<64x8x128x128xf32>
    %284 = stablehlo.multiply %283, %69 : tensor<64x8x128x128xf32>
    %285 = stablehlo.broadcast_in_dim %284, dims = [0, 1, 2, 3] : (tensor<64x8x128x128xf32>) -> tensor<64x8x128x128xf32>
    %286 = stablehlo.select %81, %82, %285 : tensor<64x8x128x128xi1>, tensor<64x8x128x128xf32>
    %287 = stablehlo.reduce(%286 init: %cst_7) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %288 = stablehlo.reshape %287 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %290 = stablehlo.subtract %286, %289 : tensor<64x8x128x128xf32>
    %291 = stablehlo.exponential %290 : tensor<64x8x128x128xf32>
    %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %293 = stablehlo.reshape %292 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %295 = stablehlo.divide %291, %294 : tensor<64x8x128x128xf32>
    %296 = stablehlo.reshape %295 : (tensor<64x8x128x128xf32>) -> tensor<512x128x128xf32>
    %297 = stablehlo.reshape %277 : (tensor<64x8x128x32xf32>) -> tensor<512x128x32xf32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2] : (tensor<512x128x32xf32>) -> tensor<512x128x32xf32>
    %299 = stablehlo.dot_general %296, %298, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x128xf32>, tensor<512x128x32xf32>) -> tensor<512x128x32xf32>
    %300 = stablehlo.reshape %299 : (tensor<512x128x32xf32>) -> tensor<64x8x128x32xf32>
    %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<64x8x128x32xf32>) -> tensor<64x128x8x32xf32>
    %302 = stablehlo.reshape %301 : (tensor<64x128x8x32xf32>) -> tensor<64x128x256xf32>
    %303 = stablehlo.transpose %arg21, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %304 = stablehlo.reshape %302 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %305 = stablehlo.dot_general %304, %303, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x256xf32>) -> tensor<8192x256xf32>
    %306 = stablehlo.reshape %305 : (tensor<8192x256xf32>) -> tensor<64x128x256xf32>
    %307 = stablehlo.add %240, %306 : tensor<64x128x256xf32>
    %308 = stablehlo.convert %307 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %309 = stablehlo.reduce(%308 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %310 = stablehlo.reshape %309 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %311 = stablehlo.divide %310, %20 : tensor<64x128x1xf64>
    %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %313 = stablehlo.subtract %308, %312 : tensor<64x128x256xf64>
    %314 = stablehlo.multiply %313, %313 : tensor<64x128x256xf64>
    %315 = stablehlo.reduce(%314 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %316 = stablehlo.reshape %315 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %317 = stablehlo.divide %316, %20 : tensor<64x128x1xf64>
    %318 = stablehlo.convert %317 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %319 = stablehlo.reduce(%307 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %320 = stablehlo.reshape %319 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %321 = stablehlo.divide %320, %33 : tensor<64x128x1xf32>
    %322 = stablehlo.add %318, %37 : tensor<64x128x1xf32>
    %323 = stablehlo.rsqrt %322 : tensor<64x128x1xf32>
    %324 = stablehlo.broadcast_in_dim %321, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %325 = stablehlo.subtract %307, %324 : tensor<64x128x256xf32>
    %326 = stablehlo.broadcast_in_dim %323, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %327 = stablehlo.multiply %325, %326 : tensor<64x128x256xf32>
    %328 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %329 = stablehlo.multiply %327, %328 : tensor<64x128x256xf32>
    %330 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %331 = stablehlo.add %329, %330 : tensor<64x128x256xf32>
    %332 = stablehlo.transpose %arg24, dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %333 = stablehlo.reshape %331 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %334 = stablehlo.dot_general %333, %332, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x1024xf32>) -> tensor<8192x1024xf32>
    %335 = stablehlo.reshape %334 : (tensor<8192x1024xf32>) -> tensor<64x128x1024xf32>
    %336 = stablehlo.maximum %335, %cst_11 : tensor<64x128x1024xf32>
    %337 = stablehlo.transpose %arg25, dims = [1, 0] : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
    %338 = stablehlo.reshape %336 : (tensor<64x128x1024xf32>) -> tensor<8192x1024xf32>
    %339 = stablehlo.dot_general %338, %337, contracting_dims = [1] x [0] : (tensor<8192x1024xf32>, tensor<1024x256xf32>) -> tensor<8192x256xf32>
    %340 = stablehlo.reshape %339 : (tensor<8192x256xf32>) -> tensor<64x128x256xf32>
    %341 = stablehlo.add %307, %340 : tensor<64x128x256xf32>
    %342 = stablehlo.convert %341 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %343 = stablehlo.reduce(%342 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %344 = stablehlo.reshape %343 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %345 = stablehlo.divide %344, %20 : tensor<64x128x1xf64>
    %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %347 = stablehlo.subtract %342, %346 : tensor<64x128x256xf64>
    %348 = stablehlo.multiply %347, %347 : tensor<64x128x256xf64>
    %349 = stablehlo.reduce(%348 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %350 = stablehlo.reshape %349 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %351 = stablehlo.divide %350, %20 : tensor<64x128x1xf64>
    %352 = stablehlo.convert %351 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %353 = stablehlo.reduce(%341 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %354 = stablehlo.reshape %353 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %355 = stablehlo.divide %354, %33 : tensor<64x128x1xf32>
    %356 = stablehlo.add %352, %37 : tensor<64x128x1xf32>
    %357 = stablehlo.rsqrt %356 : tensor<64x128x1xf32>
    %358 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %359 = stablehlo.subtract %341, %358 : tensor<64x128x256xf32>
    %360 = stablehlo.broadcast_in_dim %357, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %361 = stablehlo.multiply %359, %360 : tensor<64x128x256xf32>
    %362 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %363 = stablehlo.multiply %361, %362 : tensor<64x128x256xf32>
    %364 = stablehlo.broadcast_in_dim %arg27, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %365 = stablehlo.add %363, %364 : tensor<64x128x256xf32>
    %366 = stablehlo.transpose %arg28, dims = [1, 0] : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %367 = stablehlo.reshape %365 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %368 = stablehlo.dot_general %367, %366, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x768xf32>) -> tensor<8192x768xf32>
    %369 = stablehlo.reshape %368 : (tensor<8192x768xf32>) -> tensor<64x128x768xf32>
    %370 = stablehlo.slice %369 [0:64, 0:128, 0:256] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %371 = stablehlo.slice %369 [0:64, 0:128, 256:512] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %372 = stablehlo.slice %369 [0:64, 0:128, 512:768] : (tensor<64x128x768xf32>) -> tensor<64x128x256xf32>
    %373 = stablehlo.reshape %371 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %374 = stablehlo.transpose %373, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %375 = stablehlo.reshape %370 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %376 = stablehlo.transpose %375, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %377 = stablehlo.reshape %372 : (tensor<64x128x256xf32>) -> tensor<64x128x8x32xf32>
    %378 = stablehlo.transpose %377, dims = [0, 2, 1, 3] : (tensor<64x128x8x32xf32>) -> tensor<64x8x128x32xf32>
    %379 = stablehlo.transpose %374, dims = [0, 1, 3, 2] : (tensor<64x8x128x32xf32>) -> tensor<64x8x32x128xf32>
    %380 = stablehlo.reshape %376 : (tensor<64x8x128x32xf32>) -> tensor<512x128x32xf32>
    %381 = stablehlo.reshape %379 : (tensor<64x8x32x128xf32>) -> tensor<512x32x128xf32>
    %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1, 2] : (tensor<512x32x128xf32>) -> tensor<512x32x128xf32>
    %383 = stablehlo.dot_general %380, %382, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x32xf32>, tensor<512x32x128xf32>) -> tensor<512x128x128xf32>
    %384 = stablehlo.reshape %383 : (tensor<512x128x128xf32>) -> tensor<64x8x128x128xf32>
    %385 = stablehlo.multiply %384, %69 : tensor<64x8x128x128xf32>
    %386 = stablehlo.broadcast_in_dim %385, dims = [0, 1, 2, 3] : (tensor<64x8x128x128xf32>) -> tensor<64x8x128x128xf32>
    %387 = stablehlo.select %81, %82, %386 : tensor<64x8x128x128xi1>, tensor<64x8x128x128xf32>
    %388 = stablehlo.reduce(%387 init: %cst_7) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %389 = stablehlo.reshape %388 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %391 = stablehlo.subtract %387, %390 : tensor<64x8x128x128xf32>
    %392 = stablehlo.exponential %391 : tensor<64x8x128x128xf32>
    %393 = stablehlo.reduce(%392 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x8x128x128xf32>, tensor<f32>) -> tensor<64x8x128xf32>
    %394 = stablehlo.reshape %393 : (tensor<64x8x128xf32>) -> tensor<64x8x128x1xf32>
    %395 = stablehlo.broadcast_in_dim %394, dims = [0, 1, 2, 3] : (tensor<64x8x128x1xf32>) -> tensor<64x8x128x128xf32>
    %396 = stablehlo.divide %392, %395 : tensor<64x8x128x128xf32>
    %397 = stablehlo.reshape %396 : (tensor<64x8x128x128xf32>) -> tensor<512x128x128xf32>
    %398 = stablehlo.reshape %378 : (tensor<64x8x128x32xf32>) -> tensor<512x128x32xf32>
    %399 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2] : (tensor<512x128x32xf32>) -> tensor<512x128x32xf32>
    %400 = stablehlo.dot_general %397, %399, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<512x128x128xf32>, tensor<512x128x32xf32>) -> tensor<512x128x32xf32>
    %401 = stablehlo.reshape %400 : (tensor<512x128x32xf32>) -> tensor<64x8x128x32xf32>
    %402 = stablehlo.transpose %401, dims = [0, 2, 1, 3] : (tensor<64x8x128x32xf32>) -> tensor<64x128x8x32xf32>
    %403 = stablehlo.reshape %402 : (tensor<64x128x8x32xf32>) -> tensor<64x128x256xf32>
    %404 = stablehlo.transpose %arg29, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %405 = stablehlo.reshape %403 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %406 = stablehlo.dot_general %405, %404, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x256xf32>) -> tensor<8192x256xf32>
    %407 = stablehlo.reshape %406 : (tensor<8192x256xf32>) -> tensor<64x128x256xf32>
    %408 = stablehlo.add %341, %407 : tensor<64x128x256xf32>
    %409 = stablehlo.convert %408 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %410 = stablehlo.reduce(%409 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %411 = stablehlo.reshape %410 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %412 = stablehlo.divide %411, %20 : tensor<64x128x1xf64>
    %413 = stablehlo.broadcast_in_dim %412, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %414 = stablehlo.subtract %409, %413 : tensor<64x128x256xf64>
    %415 = stablehlo.multiply %414, %414 : tensor<64x128x256xf64>
    %416 = stablehlo.reduce(%415 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %417 = stablehlo.reshape %416 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %418 = stablehlo.divide %417, %20 : tensor<64x128x1xf64>
    %419 = stablehlo.convert %418 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %420 = stablehlo.reduce(%408 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %421 = stablehlo.reshape %420 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %422 = stablehlo.divide %421, %33 : tensor<64x128x1xf32>
    %423 = stablehlo.add %419, %37 : tensor<64x128x1xf32>
    %424 = stablehlo.rsqrt %423 : tensor<64x128x1xf32>
    %425 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %426 = stablehlo.subtract %408, %425 : tensor<64x128x256xf32>
    %427 = stablehlo.broadcast_in_dim %424, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %428 = stablehlo.multiply %426, %427 : tensor<64x128x256xf32>
    %429 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %430 = stablehlo.multiply %428, %429 : tensor<64x128x256xf32>
    %431 = stablehlo.broadcast_in_dim %arg31, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %432 = stablehlo.add %430, %431 : tensor<64x128x256xf32>
    %433 = stablehlo.transpose %arg32, dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %434 = stablehlo.reshape %432 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %435 = stablehlo.dot_general %434, %433, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x1024xf32>) -> tensor<8192x1024xf32>
    %436 = stablehlo.reshape %435 : (tensor<8192x1024xf32>) -> tensor<64x128x1024xf32>
    %437 = stablehlo.maximum %436, %cst_11 : tensor<64x128x1024xf32>
    %438 = stablehlo.transpose %arg33, dims = [1, 0] : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
    %439 = stablehlo.reshape %437 : (tensor<64x128x1024xf32>) -> tensor<8192x1024xf32>
    %440 = stablehlo.dot_general %439, %438, contracting_dims = [1] x [0] : (tensor<8192x1024xf32>, tensor<1024x256xf32>) -> tensor<8192x256xf32>
    %441 = stablehlo.reshape %440 : (tensor<8192x256xf32>) -> tensor<64x128x256xf32>
    %442 = stablehlo.add %408, %441 : tensor<64x128x256xf32>
    %443 = stablehlo.convert %442 : (tensor<64x128x256xf32>) -> tensor<64x128x256xf64>
    %444 = stablehlo.reduce(%443 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %445 = stablehlo.reshape %444 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %446 = stablehlo.divide %445, %20 : tensor<64x128x1xf64>
    %447 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2] : (tensor<64x128x1xf64>) -> tensor<64x128x256xf64>
    %448 = stablehlo.subtract %443, %447 : tensor<64x128x256xf64>
    %449 = stablehlo.multiply %448, %448 : tensor<64x128x256xf64>
    %450 = stablehlo.reduce(%449 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf64>, tensor<f64>) -> tensor<64x128xf64>
    %451 = stablehlo.reshape %450 : (tensor<64x128xf64>) -> tensor<64x128x1xf64>
    %452 = stablehlo.divide %451, %20 : tensor<64x128x1xf64>
    %453 = stablehlo.convert %452 : (tensor<64x128x1xf64>) -> tensor<64x128x1xf32>
    %454 = stablehlo.reduce(%442 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x256xf32>, tensor<f32>) -> tensor<64x128xf32>
    %455 = stablehlo.reshape %454 : (tensor<64x128xf32>) -> tensor<64x128x1xf32>
    %456 = stablehlo.divide %455, %33 : tensor<64x128x1xf32>
    %457 = stablehlo.add %453, %37 : tensor<64x128x1xf32>
    %458 = stablehlo.rsqrt %457 : tensor<64x128x1xf32>
    %459 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %460 = stablehlo.subtract %442, %459 : tensor<64x128x256xf32>
    %461 = stablehlo.broadcast_in_dim %458, dims = [0, 1, 2] : (tensor<64x128x1xf32>) -> tensor<64x128x256xf32>
    %462 = stablehlo.multiply %460, %461 : tensor<64x128x256xf32>
    %463 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %464 = stablehlo.multiply %462, %463 : tensor<64x128x256xf32>
    %465 = stablehlo.broadcast_in_dim %arg35, dims = [2] : (tensor<256xf32>) -> tensor<64x128x256xf32>
    %466 = stablehlo.add %464, %465 : tensor<64x128x256xf32>
    %467 = stablehlo.reshape %466 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %468 = stablehlo.transpose %arg36, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %469 = stablehlo.dot_general %467, %468, contracting_dims = [1] x [0] : (tensor<8192x256xf32>, tensor<256x256xf32>) -> tensor<8192x256xf32>
    %470 = stablehlo.convert %cst_12 : (tensor<1xi64>) -> tensor<1xf32>
    %471 = stablehlo.reshape %470 : (tensor<1xf32>) -> tensor<f32>
    %472 = stablehlo.broadcast_in_dim %471, dims = [] : (tensor<f32>) -> tensor<8192x256xf32>
    %473 = stablehlo.multiply %469, %472 : tensor<8192x256xf32>
    %474 = stablehlo.broadcast_in_dim %471, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %475 = stablehlo.multiply %arg37, %474 : tensor<256xf32>
    %476 = stablehlo.broadcast_in_dim %475, dims = [1] : (tensor<256xf32>) -> tensor<8192x256xf32>
    %477 = stablehlo.add %473, %476 : tensor<8192x256xf32>
    %478 = stablehlo.reshape %477 : (tensor<8192x256xf32>) -> tensor<64x128x256xf32>
    %479 = stablehlo.reshape %478 : (tensor<64x128x256xf32>) -> tensor<8192x256xf32>
    %480 = stablehlo.reshape %arg39 : (tensor<64x128xi64>) -> tensor<8192xi64>
    %481 = stablehlo.reduce(%479 init: %cst_7) applies stablehlo.maximum across dimensions = [1] : (tensor<8192x256xf32>, tensor<f32>) -> tensor<8192xf32>
    %482 = stablehlo.reshape %481 : (tensor<8192xf32>) -> tensor<8192x1xf32>
    %483 = stablehlo.broadcast_in_dim %482, dims = [0, 1] : (tensor<8192x1xf32>) -> tensor<8192x256xf32>
    %484 = stablehlo.subtract %479, %483 : tensor<8192x256xf32>
    %485 = stablehlo.exponential %484 : tensor<8192x256xf32>
    %486 = stablehlo.reduce(%485 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8192x256xf32>, tensor<f32>) -> tensor<8192xf32>
    %487 = stablehlo.reshape %486 : (tensor<8192xf32>) -> tensor<8192x1xf32>
    %488 = stablehlo.log %487 : tensor<8192x1xf32>
    %489 = stablehlo.broadcast_in_dim %488, dims = [0, 1] : (tensor<8192x1xf32>) -> tensor<8192x256xf32>
    %490 = stablehlo.subtract %484, %489 : tensor<8192x256xf32>
    %491 = stablehlo.compare  NE, %480, %c_0,  SIGNED : (tensor<8192xi64>, tensor<8192xi64>) -> tensor<8192xi1>
    %492 = stablehlo.broadcast_in_dim %491, dims = [0] : (tensor<8192xi1>) -> tensor<8192xi1>
    %493 = stablehlo.broadcast_in_dim %480, dims = [0] : (tensor<8192xi64>) -> tensor<8192xi64>
    %494 = stablehlo.select %492, %493, %c : tensor<8192xi1>, tensor<8192xi64>
    %495 = stablehlo.reshape %494 : (tensor<8192xi64>) -> tensor<8192x1xi64>
    %496 = stablehlo.iota dim = 0 : tensor<8192x1x1xi64>
    %497 = stablehlo.reshape %495 : (tensor<8192x1xi64>) -> tensor<8192x1x1xi64>
    %498 = stablehlo.concatenate %496, %497, dim = 2 : (tensor<8192x1x1xi64>, tensor<8192x1x1xi64>) -> tensor<8192x1x2xi64>
    %499 = "stablehlo.gather"(%490, %498) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<8192x256xf32>, tensor<8192x1x2xi64>) -> tensor<8192x1xf32>
    %500 = stablehlo.reshape %499 : (tensor<8192x1xf32>) -> tensor<8192xf32>
    %501 = stablehlo.negate %500 : tensor<8192xf32>
    %502 = stablehlo.broadcast_in_dim %501, dims = [0] : (tensor<8192xf32>) -> tensor<8192xf32>
    %503 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8192xf32>
    %504 = stablehlo.select %492, %502, %503 : tensor<8192xi1>, tensor<8192xf32>
    %505 = stablehlo.convert %491 : (tensor<8192xi1>) -> tensor<8192xi64>
    %506 = stablehlo.reduce(%505 init: %c_6) applies stablehlo.add across dimensions = [0] : (tensor<8192xi64>, tensor<i64>) -> tensor<i64>
    %507 = stablehlo.convert %506 : (tensor<i64>) -> tensor<f32>
    %508 = stablehlo.reduce(%504 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<8192xf32>, tensor<f32>) -> tensor<f32>
    %509 = stablehlo.divide %508, %507 : tensor<f32>
    return %509 : tensor<f32>
  }
}
