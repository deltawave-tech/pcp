module {
  func.func @main(%arg0: tensor<128x16xf32>, %arg1: tensor<128x16xf32>, %arg2: tensor<128x16xf32>, %arg3: tensor<128x16xf32>, %arg4: tensor<128x16xf32>, %arg5: tensor<128x16xf32>, %arg6: tensor<128x16xf32>, %arg7: tensor<128x16xf32>, %arg8: tensor<256x256xf32>, %arg9: tensor<256xf32>, %arg10: tensor<256xf32>, %arg11: tensor<768x256xf32>, %arg12: tensor<256x256xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<1024x256xf32>, %arg16: tensor<256x1024xf32>, %arg17: tensor<256xf32>, %arg18: tensor<256xf32>, %arg19: tensor<768x256xf32>, %arg20: tensor<256x256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<1024x256xf32>, %arg24: tensor<256x1024xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<768x256xf32>, %arg28: tensor<256x256xf32>, %arg29: tensor<256xf32>, %arg30: tensor<256xf32>, %arg31: tensor<1024x256xf32>, %arg32: tensor<256x1024xf32>, %arg33: tensor<256xf32>, %arg34: tensor<256xf32>, %arg35: tensor<768x256xf32>, %arg36: tensor<256x256xf32>, %arg37: tensor<256xf32>, %arg38: tensor<256xf32>, %arg39: tensor<1024x256xf32>, %arg40: tensor<256x1024xf32>, %arg41: tensor<256xf32>, %arg42: tensor<256xf32>, %arg43: tensor<256x256xf32>, %arg44: tensor<256xf32>, %arg45: tensor<128x16xf32>, %arg46: tensor<128x16xf32>, %arg47: tensor<32x128xi64>, %arg48: tensor<32x128xi64>) -> tensor<f32> {
    %c = stablehlo.constant dense<0> : tensor<4096xi64>
    %c_0 = stablehlo.constant dense<-100> : tensor<4096xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<128x128xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<32x128x1024xf32>
    %cst_8 = arith.constant dense<256> : tensor<1xi64>
    %cst_9 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_10 = arith.constant dense<0.17677669529663687> : tensor<1xf64>
    %cst_11 = arith.constant dense<1> : tensor<1xi64>
    %0 = "stablehlo.gather"(%arg8, %arg47) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256>}> : (tensor<256x256xf32>, tensor<32x128xi64>) -> tensor<32x128x256xf32>
    %1 = stablehlo.convert %0 : tensor<32x128x256xf32>
    %2 = stablehlo.convert %1 : (tensor<32x128x256xf32>) -> tensor<32x128x256xf64>
    %3 = stablehlo.reduce(%2 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %4 = stablehlo.reshape %3 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %5 = stablehlo.convert %cst_8 : (tensor<1xi64>) -> tensor<1xf64>
    %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f64>) -> tensor<32x128x1xf64>
    %8 = stablehlo.divide %4, %7 : tensor<32x128x1xf64>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<32x128x1xf64>) -> tensor<32x128x256xf64>
    %10 = stablehlo.subtract %2, %9 : tensor<32x128x256xf64>
    %11 = stablehlo.multiply %10, %10 : tensor<32x128x256xf64>
    %12 = stablehlo.reduce(%11 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %13 = stablehlo.reshape %12 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %14 = stablehlo.divide %13, %7 : tensor<32x128x1xf64>
    %15 = stablehlo.convert %14 : (tensor<32x128x1xf64>) -> tensor<32x128x1xf32>
    %16 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf32>, tensor<f32>) -> tensor<32x128xf32>
    %17 = stablehlo.reshape %16 : (tensor<32x128xf32>) -> tensor<32x128x1xf32>
    %18 = stablehlo.convert %cst_8 : (tensor<1xi64>) -> tensor<1xf32>
    %19 = stablehlo.reshape %18 : (tensor<1xf32>) -> tensor<f32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<32x128x1xf32>
    %21 = stablehlo.divide %17, %20 : tensor<32x128x1xf32>
    %22 = stablehlo.convert %cst_9 : (tensor<1xf64>) -> tensor<1xf32>
    %23 = stablehlo.reshape %22 : (tensor<1xf32>) -> tensor<f32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<f32>) -> tensor<32x128x1xf32>
    %25 = stablehlo.add %15, %24 : tensor<32x128x1xf32>
    %26 = stablehlo.rsqrt %25 : tensor<32x128x1xf32>
    %27 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %28 = stablehlo.subtract %1, %27 : tensor<32x128x256xf32>
    %29 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %30 = stablehlo.multiply %28, %29 : tensor<32x128x256xf32>
    %31 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<32x128x256xf32>
    %33 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %34 = stablehlo.add %32, %33 : tensor<32x128x256xf32>
    %35 = stablehlo.transpose %arg11, dims = [1, 0] : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %36 = stablehlo.reshape %34 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %37 = stablehlo.dot_general %36, %35, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x768xf32>) -> tensor<4096x768xf32>
    %38 = stablehlo.reshape %37 : (tensor<4096x768xf32>) -> tensor<32x128x768xf32>
    %39 = stablehlo.slice %38 [0:32, 0:128, 0:256] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %40 = stablehlo.slice %38 [0:32, 0:128, 256:512] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %41 = stablehlo.slice %38 [0:32, 0:128, 512:768] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %42 = stablehlo.reshape %39 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %43 = stablehlo.reshape %40 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %44 = stablehlo.reshape %41 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %45 = stablehlo.transpose %44, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %46 = stablehlo.slice %42 [0:32, 0:128, 0:8, 0:16] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %47 = stablehlo.slice %42 [0:32, 0:128, 0:8, 16:32] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %48 = stablehlo.reshape %arg45 : (tensor<128x16xf32>) -> tensor<1x128x1x16xf32>
    %49 = stablehlo.reshape %arg46 : (tensor<128x16xf32>) -> tensor<1x128x1x16xf32>
    %50 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<1x128x1x16xf32>) -> tensor<32x128x8x16xf32>
    %51 = stablehlo.multiply %46, %50 : tensor<32x128x8x16xf32>
    %52 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2, 3] : (tensor<1x128x1x16xf32>) -> tensor<32x128x8x16xf32>
    %53 = stablehlo.multiply %47, %52 : tensor<32x128x8x16xf32>
    %54 = stablehlo.subtract %51, %53 : tensor<32x128x8x16xf32>
    %55 = stablehlo.multiply %47, %50 : tensor<32x128x8x16xf32>
    %56 = stablehlo.multiply %46, %52 : tensor<32x128x8x16xf32>
    %57 = stablehlo.add %55, %56 : tensor<32x128x8x16xf32>
    %58 = stablehlo.concatenate %54, %57, dim = 3 : (tensor<32x128x8x16xf32>, tensor<32x128x8x16xf32>) -> tensor<32x128x8x32xf32>
    %59 = stablehlo.transpose %58, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %60 = stablehlo.slice %43 [0:32, 0:128, 0:8, 0:16] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %61 = stablehlo.slice %43 [0:32, 0:128, 0:8, 16:32] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %62 = stablehlo.multiply %60, %50 : tensor<32x128x8x16xf32>
    %63 = stablehlo.multiply %61, %52 : tensor<32x128x8x16xf32>
    %64 = stablehlo.subtract %62, %63 : tensor<32x128x8x16xf32>
    %65 = stablehlo.multiply %61, %50 : tensor<32x128x8x16xf32>
    %66 = stablehlo.multiply %60, %52 : tensor<32x128x8x16xf32>
    %67 = stablehlo.add %65, %66 : tensor<32x128x8x16xf32>
    %68 = stablehlo.concatenate %64, %67, dim = 3 : (tensor<32x128x8x16xf32>, tensor<32x128x8x16xf32>) -> tensor<32x128x8x32xf32>
    %69 = stablehlo.transpose %68, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %70 = stablehlo.transpose %69, dims = [0, 1, 3, 2] : (tensor<32x8x128x32xf32>) -> tensor<32x8x32x128xf32>
    %71 = stablehlo.reshape %59 : (tensor<32x8x128x32xf32>) -> tensor<256x128x32xf32>
    %72 = stablehlo.reshape %70 : (tensor<32x8x32x128xf32>) -> tensor<256x32x128xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2] : (tensor<256x32x128xf32>) -> tensor<256x32x128xf32>
    %74 = stablehlo.dot_general %71, %73, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x32xf32>, tensor<256x32x128xf32>) -> tensor<256x128x128xf32>
    %75 = stablehlo.reshape %74 : (tensor<256x128x128xf32>) -> tensor<32x8x128x128xf32>
    %76 = stablehlo.convert %cst_10 : (tensor<1xf64>) -> tensor<1xf32>
    %77 = stablehlo.reshape %76 : (tensor<1xf32>) -> tensor<f32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [] : (tensor<f32>) -> tensor<32x8x128x128xf32>
    %79 = stablehlo.multiply %75, %78 : tensor<32x8x128x128xf32>
    %80 = stablehlo.iota dim = 1 : tensor<128x128xi64>
    %81 = stablehlo.iota dim = 0 : tensor<128x128xi64>
    %82 = stablehlo.add %81, %c_1 : tensor<128x128xi64>
    %83 = stablehlo.compare  LE, %80, %82,  SIGNED : (tensor<128x128xi64>, tensor<128x128xi64>) -> tensor<128x128xi1>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1] : (tensor<128x128xi1>) -> tensor<128x128xi1>
    %85 = stablehlo.select %84, %cst_4, %cst_6 : tensor<128x128xi1>, tensor<128x128xf32>
    %86 = stablehlo.reshape %85 : (tensor<128x128xf32>) -> tensor<1x1x128x128xf32>
    %87 = stablehlo.convert %c_2 : (tensor<i64>) -> tensor<f32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [] : (tensor<f32>) -> tensor<1x1x128x128xf32>
    %89 = stablehlo.compare  EQ, %86, %88,  FLOAT : (tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xi1>
    %90 = stablehlo.broadcast_in_dim %89, dims = [0, 1, 2, 3] : (tensor<1x1x128x128xi1>) -> tensor<32x8x128x128xi1>
    %91 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<32x8x128x128xf32>
    %92 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2, 3] : (tensor<32x8x128x128xf32>) -> tensor<32x8x128x128xf32>
    %93 = stablehlo.select %90, %91, %92 : tensor<32x8x128x128xi1>, tensor<32x8x128x128xf32>
    %94 = stablehlo.reduce(%93 init: %cst_3) applies stablehlo.maximum across dimensions = [3] : (tensor<32x8x128x128xf32>, tensor<f32>) -> tensor<32x8x128xf32>
    %95 = stablehlo.reshape %94 : (tensor<32x8x128xf32>) -> tensor<32x8x128x1xf32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<32x8x128x1xf32>) -> tensor<32x8x128x128xf32>
    %97 = stablehlo.subtract %93, %96 : tensor<32x8x128x128xf32>
    %98 = stablehlo.exponential %97 : tensor<32x8x128x128xf32>
    %99 = stablehlo.reduce(%98 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<32x8x128x128xf32>, tensor<f32>) -> tensor<32x8x128xf32>
    %100 = stablehlo.reshape %99 : (tensor<32x8x128xf32>) -> tensor<32x8x128x1xf32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2, 3] : (tensor<32x8x128x1xf32>) -> tensor<32x8x128x128xf32>
    %102 = stablehlo.divide %98, %101 : tensor<32x8x128x128xf32>
    %103 = stablehlo.reshape %102 : (tensor<32x8x128x128xf32>) -> tensor<256x128x128xf32>
    %104 = stablehlo.reshape %45 : (tensor<32x8x128x32xf32>) -> tensor<256x128x32xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2] : (tensor<256x128x32xf32>) -> tensor<256x128x32xf32>
    %106 = stablehlo.dot_general %103, %105, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x128xf32>, tensor<256x128x32xf32>) -> tensor<256x128x32xf32>
    %107 = stablehlo.reshape %106 : (tensor<256x128x32xf32>) -> tensor<32x8x128x32xf32>
    %108 = stablehlo.transpose %107, dims = [0, 2, 1, 3] : (tensor<32x8x128x32xf32>) -> tensor<32x128x8x32xf32>
    %109 = stablehlo.reshape %108 : (tensor<32x128x8x32xf32>) -> tensor<32x128x256xf32>
    %110 = stablehlo.transpose %arg12, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %111 = stablehlo.reshape %109 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %112 = stablehlo.dot_general %111, %110, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %113 = stablehlo.reshape %112 : (tensor<4096x256xf32>) -> tensor<32x128x256xf32>
    %114 = stablehlo.add %1, %113 : tensor<32x128x256xf32>
    %115 = stablehlo.convert %114 : (tensor<32x128x256xf32>) -> tensor<32x128x256xf64>
    %116 = stablehlo.reduce(%115 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %117 = stablehlo.reshape %116 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %118 = stablehlo.divide %117, %7 : tensor<32x128x1xf64>
    %119 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2] : (tensor<32x128x1xf64>) -> tensor<32x128x256xf64>
    %120 = stablehlo.subtract %115, %119 : tensor<32x128x256xf64>
    %121 = stablehlo.multiply %120, %120 : tensor<32x128x256xf64>
    %122 = stablehlo.reduce(%121 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %123 = stablehlo.reshape %122 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %124 = stablehlo.divide %123, %7 : tensor<32x128x1xf64>
    %125 = stablehlo.convert %124 : (tensor<32x128x1xf64>) -> tensor<32x128x1xf32>
    %126 = stablehlo.reduce(%114 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf32>, tensor<f32>) -> tensor<32x128xf32>
    %127 = stablehlo.reshape %126 : (tensor<32x128xf32>) -> tensor<32x128x1xf32>
    %128 = stablehlo.divide %127, %20 : tensor<32x128x1xf32>
    %129 = stablehlo.add %125, %24 : tensor<32x128x1xf32>
    %130 = stablehlo.rsqrt %129 : tensor<32x128x1xf32>
    %131 = stablehlo.broadcast_in_dim %128, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %132 = stablehlo.subtract %114, %131 : tensor<32x128x256xf32>
    %133 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %134 = stablehlo.multiply %132, %133 : tensor<32x128x256xf32>
    %135 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %136 = stablehlo.multiply %134, %135 : tensor<32x128x256xf32>
    %137 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %138 = stablehlo.add %136, %137 : tensor<32x128x256xf32>
    %139 = stablehlo.transpose %arg15, dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %140 = stablehlo.reshape %138 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %141 = stablehlo.dot_general %140, %139, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x1024xf32>) -> tensor<4096x1024xf32>
    %142 = stablehlo.reshape %141 : (tensor<4096x1024xf32>) -> tensor<32x128x1024xf32>
    %143 = stablehlo.maximum %142, %cst_7 : tensor<32x128x1024xf32>
    %144 = stablehlo.transpose %arg16, dims = [1, 0] : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
    %145 = stablehlo.reshape %143 : (tensor<32x128x1024xf32>) -> tensor<4096x1024xf32>
    %146 = stablehlo.dot_general %145, %144, contracting_dims = [1] x [0] : (tensor<4096x1024xf32>, tensor<1024x256xf32>) -> tensor<4096x256xf32>
    %147 = stablehlo.reshape %146 : (tensor<4096x256xf32>) -> tensor<32x128x256xf32>
    %148 = stablehlo.add %114, %147 : tensor<32x128x256xf32>
    %149 = stablehlo.convert %148 : (tensor<32x128x256xf32>) -> tensor<32x128x256xf64>
    %150 = stablehlo.reduce(%149 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %151 = stablehlo.reshape %150 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %152 = stablehlo.divide %151, %7 : tensor<32x128x1xf64>
    %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2] : (tensor<32x128x1xf64>) -> tensor<32x128x256xf64>
    %154 = stablehlo.subtract %149, %153 : tensor<32x128x256xf64>
    %155 = stablehlo.multiply %154, %154 : tensor<32x128x256xf64>
    %156 = stablehlo.reduce(%155 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %157 = stablehlo.reshape %156 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %158 = stablehlo.divide %157, %7 : tensor<32x128x1xf64>
    %159 = stablehlo.convert %158 : (tensor<32x128x1xf64>) -> tensor<32x128x1xf32>
    %160 = stablehlo.reduce(%148 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf32>, tensor<f32>) -> tensor<32x128xf32>
    %161 = stablehlo.reshape %160 : (tensor<32x128xf32>) -> tensor<32x128x1xf32>
    %162 = stablehlo.divide %161, %20 : tensor<32x128x1xf32>
    %163 = stablehlo.add %159, %24 : tensor<32x128x1xf32>
    %164 = stablehlo.rsqrt %163 : tensor<32x128x1xf32>
    %165 = stablehlo.broadcast_in_dim %162, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %166 = stablehlo.subtract %148, %165 : tensor<32x128x256xf32>
    %167 = stablehlo.broadcast_in_dim %164, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %168 = stablehlo.multiply %166, %167 : tensor<32x128x256xf32>
    %169 = stablehlo.broadcast_in_dim %arg17, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %170 = stablehlo.multiply %168, %169 : tensor<32x128x256xf32>
    %171 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %172 = stablehlo.add %170, %171 : tensor<32x128x256xf32>
    %173 = stablehlo.transpose %arg19, dims = [1, 0] : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %174 = stablehlo.reshape %172 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %175 = stablehlo.dot_general %174, %173, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x768xf32>) -> tensor<4096x768xf32>
    %176 = stablehlo.reshape %175 : (tensor<4096x768xf32>) -> tensor<32x128x768xf32>
    %177 = stablehlo.slice %176 [0:32, 0:128, 0:256] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %178 = stablehlo.slice %176 [0:32, 0:128, 256:512] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %179 = stablehlo.slice %176 [0:32, 0:128, 512:768] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %180 = stablehlo.reshape %177 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %181 = stablehlo.reshape %178 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %182 = stablehlo.reshape %179 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %183 = stablehlo.transpose %182, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %184 = stablehlo.slice %180 [0:32, 0:128, 0:8, 0:16] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %185 = stablehlo.slice %180 [0:32, 0:128, 0:8, 16:32] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %186 = stablehlo.reshape %arg2 : (tensor<128x16xf32>) -> tensor<1x128x1x16xf32>
    %187 = stablehlo.reshape %arg3 : (tensor<128x16xf32>) -> tensor<1x128x1x16xf32>
    %188 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2, 3] : (tensor<1x128x1x16xf32>) -> tensor<32x128x8x16xf32>
    %189 = stablehlo.multiply %184, %188 : tensor<32x128x8x16xf32>
    %190 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<1x128x1x16xf32>) -> tensor<32x128x8x16xf32>
    %191 = stablehlo.multiply %185, %190 : tensor<32x128x8x16xf32>
    %192 = stablehlo.subtract %189, %191 : tensor<32x128x8x16xf32>
    %193 = stablehlo.multiply %185, %188 : tensor<32x128x8x16xf32>
    %194 = stablehlo.multiply %184, %190 : tensor<32x128x8x16xf32>
    %195 = stablehlo.add %193, %194 : tensor<32x128x8x16xf32>
    %196 = stablehlo.concatenate %192, %195, dim = 3 : (tensor<32x128x8x16xf32>, tensor<32x128x8x16xf32>) -> tensor<32x128x8x32xf32>
    %197 = stablehlo.transpose %196, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %198 = stablehlo.slice %181 [0:32, 0:128, 0:8, 0:16] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %199 = stablehlo.slice %181 [0:32, 0:128, 0:8, 16:32] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %200 = stablehlo.multiply %198, %188 : tensor<32x128x8x16xf32>
    %201 = stablehlo.multiply %199, %190 : tensor<32x128x8x16xf32>
    %202 = stablehlo.subtract %200, %201 : tensor<32x128x8x16xf32>
    %203 = stablehlo.multiply %199, %188 : tensor<32x128x8x16xf32>
    %204 = stablehlo.multiply %198, %190 : tensor<32x128x8x16xf32>
    %205 = stablehlo.add %203, %204 : tensor<32x128x8x16xf32>
    %206 = stablehlo.concatenate %202, %205, dim = 3 : (tensor<32x128x8x16xf32>, tensor<32x128x8x16xf32>) -> tensor<32x128x8x32xf32>
    %207 = stablehlo.transpose %206, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %208 = stablehlo.transpose %207, dims = [0, 1, 3, 2] : (tensor<32x8x128x32xf32>) -> tensor<32x8x32x128xf32>
    %209 = stablehlo.reshape %197 : (tensor<32x8x128x32xf32>) -> tensor<256x128x32xf32>
    %210 = stablehlo.reshape %208 : (tensor<32x8x32x128xf32>) -> tensor<256x32x128xf32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<256x32x128xf32>) -> tensor<256x32x128xf32>
    %212 = stablehlo.dot_general %209, %211, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x32xf32>, tensor<256x32x128xf32>) -> tensor<256x128x128xf32>
    %213 = stablehlo.reshape %212 : (tensor<256x128x128xf32>) -> tensor<32x8x128x128xf32>
    %214 = stablehlo.multiply %213, %78 : tensor<32x8x128x128xf32>
    %215 = stablehlo.broadcast_in_dim %214, dims = [0, 1, 2, 3] : (tensor<32x8x128x128xf32>) -> tensor<32x8x128x128xf32>
    %216 = stablehlo.select %90, %91, %215 : tensor<32x8x128x128xi1>, tensor<32x8x128x128xf32>
    %217 = stablehlo.reduce(%216 init: %cst_3) applies stablehlo.maximum across dimensions = [3] : (tensor<32x8x128x128xf32>, tensor<f32>) -> tensor<32x8x128xf32>
    %218 = stablehlo.reshape %217 : (tensor<32x8x128xf32>) -> tensor<32x8x128x1xf32>
    %219 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2, 3] : (tensor<32x8x128x1xf32>) -> tensor<32x8x128x128xf32>
    %220 = stablehlo.subtract %216, %219 : tensor<32x8x128x128xf32>
    %221 = stablehlo.exponential %220 : tensor<32x8x128x128xf32>
    %222 = stablehlo.reduce(%221 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<32x8x128x128xf32>, tensor<f32>) -> tensor<32x8x128xf32>
    %223 = stablehlo.reshape %222 : (tensor<32x8x128xf32>) -> tensor<32x8x128x1xf32>
    %224 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2, 3] : (tensor<32x8x128x1xf32>) -> tensor<32x8x128x128xf32>
    %225 = stablehlo.divide %221, %224 : tensor<32x8x128x128xf32>
    %226 = stablehlo.reshape %225 : (tensor<32x8x128x128xf32>) -> tensor<256x128x128xf32>
    %227 = stablehlo.reshape %183 : (tensor<32x8x128x32xf32>) -> tensor<256x128x32xf32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2] : (tensor<256x128x32xf32>) -> tensor<256x128x32xf32>
    %229 = stablehlo.dot_general %226, %228, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x128xf32>, tensor<256x128x32xf32>) -> tensor<256x128x32xf32>
    %230 = stablehlo.reshape %229 : (tensor<256x128x32xf32>) -> tensor<32x8x128x32xf32>
    %231 = stablehlo.transpose %230, dims = [0, 2, 1, 3] : (tensor<32x8x128x32xf32>) -> tensor<32x128x8x32xf32>
    %232 = stablehlo.reshape %231 : (tensor<32x128x8x32xf32>) -> tensor<32x128x256xf32>
    %233 = stablehlo.transpose %arg20, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %234 = stablehlo.reshape %232 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %235 = stablehlo.dot_general %234, %233, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %236 = stablehlo.reshape %235 : (tensor<4096x256xf32>) -> tensor<32x128x256xf32>
    %237 = stablehlo.add %148, %236 : tensor<32x128x256xf32>
    %238 = stablehlo.convert %237 : (tensor<32x128x256xf32>) -> tensor<32x128x256xf64>
    %239 = stablehlo.reduce(%238 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %240 = stablehlo.reshape %239 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %241 = stablehlo.divide %240, %7 : tensor<32x128x1xf64>
    %242 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2] : (tensor<32x128x1xf64>) -> tensor<32x128x256xf64>
    %243 = stablehlo.subtract %238, %242 : tensor<32x128x256xf64>
    %244 = stablehlo.multiply %243, %243 : tensor<32x128x256xf64>
    %245 = stablehlo.reduce(%244 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %246 = stablehlo.reshape %245 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %247 = stablehlo.divide %246, %7 : tensor<32x128x1xf64>
    %248 = stablehlo.convert %247 : (tensor<32x128x1xf64>) -> tensor<32x128x1xf32>
    %249 = stablehlo.reduce(%237 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf32>, tensor<f32>) -> tensor<32x128xf32>
    %250 = stablehlo.reshape %249 : (tensor<32x128xf32>) -> tensor<32x128x1xf32>
    %251 = stablehlo.divide %250, %20 : tensor<32x128x1xf32>
    %252 = stablehlo.add %248, %24 : tensor<32x128x1xf32>
    %253 = stablehlo.rsqrt %252 : tensor<32x128x1xf32>
    %254 = stablehlo.broadcast_in_dim %251, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %255 = stablehlo.subtract %237, %254 : tensor<32x128x256xf32>
    %256 = stablehlo.broadcast_in_dim %253, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %257 = stablehlo.multiply %255, %256 : tensor<32x128x256xf32>
    %258 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %259 = stablehlo.multiply %257, %258 : tensor<32x128x256xf32>
    %260 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %261 = stablehlo.add %259, %260 : tensor<32x128x256xf32>
    %262 = stablehlo.transpose %arg23, dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %263 = stablehlo.reshape %261 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %264 = stablehlo.dot_general %263, %262, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x1024xf32>) -> tensor<4096x1024xf32>
    %265 = stablehlo.reshape %264 : (tensor<4096x1024xf32>) -> tensor<32x128x1024xf32>
    %266 = stablehlo.maximum %265, %cst_7 : tensor<32x128x1024xf32>
    %267 = stablehlo.transpose %arg24, dims = [1, 0] : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
    %268 = stablehlo.reshape %266 : (tensor<32x128x1024xf32>) -> tensor<4096x1024xf32>
    %269 = stablehlo.dot_general %268, %267, contracting_dims = [1] x [0] : (tensor<4096x1024xf32>, tensor<1024x256xf32>) -> tensor<4096x256xf32>
    %270 = stablehlo.reshape %269 : (tensor<4096x256xf32>) -> tensor<32x128x256xf32>
    %271 = stablehlo.add %237, %270 : tensor<32x128x256xf32>
    %272 = stablehlo.convert %271 : (tensor<32x128x256xf32>) -> tensor<32x128x256xf64>
    %273 = stablehlo.reduce(%272 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %274 = stablehlo.reshape %273 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %275 = stablehlo.divide %274, %7 : tensor<32x128x1xf64>
    %276 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2] : (tensor<32x128x1xf64>) -> tensor<32x128x256xf64>
    %277 = stablehlo.subtract %272, %276 : tensor<32x128x256xf64>
    %278 = stablehlo.multiply %277, %277 : tensor<32x128x256xf64>
    %279 = stablehlo.reduce(%278 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %280 = stablehlo.reshape %279 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %281 = stablehlo.divide %280, %7 : tensor<32x128x1xf64>
    %282 = stablehlo.convert %281 : (tensor<32x128x1xf64>) -> tensor<32x128x1xf32>
    %283 = stablehlo.reduce(%271 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf32>, tensor<f32>) -> tensor<32x128xf32>
    %284 = stablehlo.reshape %283 : (tensor<32x128xf32>) -> tensor<32x128x1xf32>
    %285 = stablehlo.divide %284, %20 : tensor<32x128x1xf32>
    %286 = stablehlo.add %282, %24 : tensor<32x128x1xf32>
    %287 = stablehlo.rsqrt %286 : tensor<32x128x1xf32>
    %288 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %289 = stablehlo.subtract %271, %288 : tensor<32x128x256xf32>
    %290 = stablehlo.broadcast_in_dim %287, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %291 = stablehlo.multiply %289, %290 : tensor<32x128x256xf32>
    %292 = stablehlo.broadcast_in_dim %arg25, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %293 = stablehlo.multiply %291, %292 : tensor<32x128x256xf32>
    %294 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %295 = stablehlo.add %293, %294 : tensor<32x128x256xf32>
    %296 = stablehlo.transpose %arg27, dims = [1, 0] : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %297 = stablehlo.reshape %295 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %298 = stablehlo.dot_general %297, %296, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x768xf32>) -> tensor<4096x768xf32>
    %299 = stablehlo.reshape %298 : (tensor<4096x768xf32>) -> tensor<32x128x768xf32>
    %300 = stablehlo.slice %299 [0:32, 0:128, 0:256] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %301 = stablehlo.slice %299 [0:32, 0:128, 256:512] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %302 = stablehlo.slice %299 [0:32, 0:128, 512:768] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %303 = stablehlo.reshape %300 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %304 = stablehlo.reshape %301 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %305 = stablehlo.reshape %302 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %307 = stablehlo.slice %303 [0:32, 0:128, 0:8, 0:16] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %308 = stablehlo.slice %303 [0:32, 0:128, 0:8, 16:32] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %309 = stablehlo.reshape %arg4 : (tensor<128x16xf32>) -> tensor<1x128x1x16xf32>
    %310 = stablehlo.reshape %arg5 : (tensor<128x16xf32>) -> tensor<1x128x1x16xf32>
    %311 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2, 3] : (tensor<1x128x1x16xf32>) -> tensor<32x128x8x16xf32>
    %312 = stablehlo.multiply %307, %311 : tensor<32x128x8x16xf32>
    %313 = stablehlo.broadcast_in_dim %310, dims = [0, 1, 2, 3] : (tensor<1x128x1x16xf32>) -> tensor<32x128x8x16xf32>
    %314 = stablehlo.multiply %308, %313 : tensor<32x128x8x16xf32>
    %315 = stablehlo.subtract %312, %314 : tensor<32x128x8x16xf32>
    %316 = stablehlo.multiply %308, %311 : tensor<32x128x8x16xf32>
    %317 = stablehlo.multiply %307, %313 : tensor<32x128x8x16xf32>
    %318 = stablehlo.add %316, %317 : tensor<32x128x8x16xf32>
    %319 = stablehlo.concatenate %315, %318, dim = 3 : (tensor<32x128x8x16xf32>, tensor<32x128x8x16xf32>) -> tensor<32x128x8x32xf32>
    %320 = stablehlo.transpose %319, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %321 = stablehlo.slice %304 [0:32, 0:128, 0:8, 0:16] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %322 = stablehlo.slice %304 [0:32, 0:128, 0:8, 16:32] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %323 = stablehlo.multiply %321, %311 : tensor<32x128x8x16xf32>
    %324 = stablehlo.multiply %322, %313 : tensor<32x128x8x16xf32>
    %325 = stablehlo.subtract %323, %324 : tensor<32x128x8x16xf32>
    %326 = stablehlo.multiply %322, %311 : tensor<32x128x8x16xf32>
    %327 = stablehlo.multiply %321, %313 : tensor<32x128x8x16xf32>
    %328 = stablehlo.add %326, %327 : tensor<32x128x8x16xf32>
    %329 = stablehlo.concatenate %325, %328, dim = 3 : (tensor<32x128x8x16xf32>, tensor<32x128x8x16xf32>) -> tensor<32x128x8x32xf32>
    %330 = stablehlo.transpose %329, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %331 = stablehlo.transpose %330, dims = [0, 1, 3, 2] : (tensor<32x8x128x32xf32>) -> tensor<32x8x32x128xf32>
    %332 = stablehlo.reshape %320 : (tensor<32x8x128x32xf32>) -> tensor<256x128x32xf32>
    %333 = stablehlo.reshape %331 : (tensor<32x8x32x128xf32>) -> tensor<256x32x128xf32>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1, 2] : (tensor<256x32x128xf32>) -> tensor<256x32x128xf32>
    %335 = stablehlo.dot_general %332, %334, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x32xf32>, tensor<256x32x128xf32>) -> tensor<256x128x128xf32>
    %336 = stablehlo.reshape %335 : (tensor<256x128x128xf32>) -> tensor<32x8x128x128xf32>
    %337 = stablehlo.multiply %336, %78 : tensor<32x8x128x128xf32>
    %338 = stablehlo.broadcast_in_dim %337, dims = [0, 1, 2, 3] : (tensor<32x8x128x128xf32>) -> tensor<32x8x128x128xf32>
    %339 = stablehlo.select %90, %91, %338 : tensor<32x8x128x128xi1>, tensor<32x8x128x128xf32>
    %340 = stablehlo.reduce(%339 init: %cst_3) applies stablehlo.maximum across dimensions = [3] : (tensor<32x8x128x128xf32>, tensor<f32>) -> tensor<32x8x128xf32>
    %341 = stablehlo.reshape %340 : (tensor<32x8x128xf32>) -> tensor<32x8x128x1xf32>
    %342 = stablehlo.broadcast_in_dim %341, dims = [0, 1, 2, 3] : (tensor<32x8x128x1xf32>) -> tensor<32x8x128x128xf32>
    %343 = stablehlo.subtract %339, %342 : tensor<32x8x128x128xf32>
    %344 = stablehlo.exponential %343 : tensor<32x8x128x128xf32>
    %345 = stablehlo.reduce(%344 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<32x8x128x128xf32>, tensor<f32>) -> tensor<32x8x128xf32>
    %346 = stablehlo.reshape %345 : (tensor<32x8x128xf32>) -> tensor<32x8x128x1xf32>
    %347 = stablehlo.broadcast_in_dim %346, dims = [0, 1, 2, 3] : (tensor<32x8x128x1xf32>) -> tensor<32x8x128x128xf32>
    %348 = stablehlo.divide %344, %347 : tensor<32x8x128x128xf32>
    %349 = stablehlo.reshape %348 : (tensor<32x8x128x128xf32>) -> tensor<256x128x128xf32>
    %350 = stablehlo.reshape %306 : (tensor<32x8x128x32xf32>) -> tensor<256x128x32xf32>
    %351 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2] : (tensor<256x128x32xf32>) -> tensor<256x128x32xf32>
    %352 = stablehlo.dot_general %349, %351, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x128xf32>, tensor<256x128x32xf32>) -> tensor<256x128x32xf32>
    %353 = stablehlo.reshape %352 : (tensor<256x128x32xf32>) -> tensor<32x8x128x32xf32>
    %354 = stablehlo.transpose %353, dims = [0, 2, 1, 3] : (tensor<32x8x128x32xf32>) -> tensor<32x128x8x32xf32>
    %355 = stablehlo.reshape %354 : (tensor<32x128x8x32xf32>) -> tensor<32x128x256xf32>
    %356 = stablehlo.transpose %arg28, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %357 = stablehlo.reshape %355 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %358 = stablehlo.dot_general %357, %356, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %359 = stablehlo.reshape %358 : (tensor<4096x256xf32>) -> tensor<32x128x256xf32>
    %360 = stablehlo.add %271, %359 : tensor<32x128x256xf32>
    %361 = stablehlo.convert %360 : (tensor<32x128x256xf32>) -> tensor<32x128x256xf64>
    %362 = stablehlo.reduce(%361 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %363 = stablehlo.reshape %362 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %364 = stablehlo.divide %363, %7 : tensor<32x128x1xf64>
    %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<32x128x1xf64>) -> tensor<32x128x256xf64>
    %366 = stablehlo.subtract %361, %365 : tensor<32x128x256xf64>
    %367 = stablehlo.multiply %366, %366 : tensor<32x128x256xf64>
    %368 = stablehlo.reduce(%367 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %369 = stablehlo.reshape %368 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %370 = stablehlo.divide %369, %7 : tensor<32x128x1xf64>
    %371 = stablehlo.convert %370 : (tensor<32x128x1xf64>) -> tensor<32x128x1xf32>
    %372 = stablehlo.reduce(%360 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf32>, tensor<f32>) -> tensor<32x128xf32>
    %373 = stablehlo.reshape %372 : (tensor<32x128xf32>) -> tensor<32x128x1xf32>
    %374 = stablehlo.divide %373, %20 : tensor<32x128x1xf32>
    %375 = stablehlo.add %371, %24 : tensor<32x128x1xf32>
    %376 = stablehlo.rsqrt %375 : tensor<32x128x1xf32>
    %377 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %378 = stablehlo.subtract %360, %377 : tensor<32x128x256xf32>
    %379 = stablehlo.broadcast_in_dim %376, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %380 = stablehlo.multiply %378, %379 : tensor<32x128x256xf32>
    %381 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %382 = stablehlo.multiply %380, %381 : tensor<32x128x256xf32>
    %383 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %384 = stablehlo.add %382, %383 : tensor<32x128x256xf32>
    %385 = stablehlo.transpose %arg31, dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %386 = stablehlo.reshape %384 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %387 = stablehlo.dot_general %386, %385, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x1024xf32>) -> tensor<4096x1024xf32>
    %388 = stablehlo.reshape %387 : (tensor<4096x1024xf32>) -> tensor<32x128x1024xf32>
    %389 = stablehlo.maximum %388, %cst_7 : tensor<32x128x1024xf32>
    %390 = stablehlo.transpose %arg32, dims = [1, 0] : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
    %391 = stablehlo.reshape %389 : (tensor<32x128x1024xf32>) -> tensor<4096x1024xf32>
    %392 = stablehlo.dot_general %391, %390, contracting_dims = [1] x [0] : (tensor<4096x1024xf32>, tensor<1024x256xf32>) -> tensor<4096x256xf32>
    %393 = stablehlo.reshape %392 : (tensor<4096x256xf32>) -> tensor<32x128x256xf32>
    %394 = stablehlo.add %360, %393 : tensor<32x128x256xf32>
    %395 = stablehlo.convert %394 : (tensor<32x128x256xf32>) -> tensor<32x128x256xf64>
    %396 = stablehlo.reduce(%395 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %397 = stablehlo.reshape %396 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %398 = stablehlo.divide %397, %7 : tensor<32x128x1xf64>
    %399 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2] : (tensor<32x128x1xf64>) -> tensor<32x128x256xf64>
    %400 = stablehlo.subtract %395, %399 : tensor<32x128x256xf64>
    %401 = stablehlo.multiply %400, %400 : tensor<32x128x256xf64>
    %402 = stablehlo.reduce(%401 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %403 = stablehlo.reshape %402 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %404 = stablehlo.divide %403, %7 : tensor<32x128x1xf64>
    %405 = stablehlo.convert %404 : (tensor<32x128x1xf64>) -> tensor<32x128x1xf32>
    %406 = stablehlo.reduce(%394 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf32>, tensor<f32>) -> tensor<32x128xf32>
    %407 = stablehlo.reshape %406 : (tensor<32x128xf32>) -> tensor<32x128x1xf32>
    %408 = stablehlo.divide %407, %20 : tensor<32x128x1xf32>
    %409 = stablehlo.add %405, %24 : tensor<32x128x1xf32>
    %410 = stablehlo.rsqrt %409 : tensor<32x128x1xf32>
    %411 = stablehlo.broadcast_in_dim %408, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %412 = stablehlo.subtract %394, %411 : tensor<32x128x256xf32>
    %413 = stablehlo.broadcast_in_dim %410, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %414 = stablehlo.multiply %412, %413 : tensor<32x128x256xf32>
    %415 = stablehlo.broadcast_in_dim %arg33, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %416 = stablehlo.multiply %414, %415 : tensor<32x128x256xf32>
    %417 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %418 = stablehlo.add %416, %417 : tensor<32x128x256xf32>
    %419 = stablehlo.transpose %arg35, dims = [1, 0] : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %420 = stablehlo.reshape %418 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %421 = stablehlo.dot_general %420, %419, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x768xf32>) -> tensor<4096x768xf32>
    %422 = stablehlo.reshape %421 : (tensor<4096x768xf32>) -> tensor<32x128x768xf32>
    %423 = stablehlo.slice %422 [0:32, 0:128, 0:256] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %424 = stablehlo.slice %422 [0:32, 0:128, 256:512] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %425 = stablehlo.slice %422 [0:32, 0:128, 512:768] : (tensor<32x128x768xf32>) -> tensor<32x128x256xf32>
    %426 = stablehlo.reshape %423 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %427 = stablehlo.reshape %424 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %428 = stablehlo.reshape %425 : (tensor<32x128x256xf32>) -> tensor<32x128x8x32xf32>
    %429 = stablehlo.transpose %428, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %430 = stablehlo.slice %426 [0:32, 0:128, 0:8, 0:16] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %431 = stablehlo.slice %426 [0:32, 0:128, 0:8, 16:32] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %432 = stablehlo.reshape %arg6 : (tensor<128x16xf32>) -> tensor<1x128x1x16xf32>
    %433 = stablehlo.reshape %arg7 : (tensor<128x16xf32>) -> tensor<1x128x1x16xf32>
    %434 = stablehlo.broadcast_in_dim %432, dims = [0, 1, 2, 3] : (tensor<1x128x1x16xf32>) -> tensor<32x128x8x16xf32>
    %435 = stablehlo.multiply %430, %434 : tensor<32x128x8x16xf32>
    %436 = stablehlo.broadcast_in_dim %433, dims = [0, 1, 2, 3] : (tensor<1x128x1x16xf32>) -> tensor<32x128x8x16xf32>
    %437 = stablehlo.multiply %431, %436 : tensor<32x128x8x16xf32>
    %438 = stablehlo.subtract %435, %437 : tensor<32x128x8x16xf32>
    %439 = stablehlo.multiply %431, %434 : tensor<32x128x8x16xf32>
    %440 = stablehlo.multiply %430, %436 : tensor<32x128x8x16xf32>
    %441 = stablehlo.add %439, %440 : tensor<32x128x8x16xf32>
    %442 = stablehlo.concatenate %438, %441, dim = 3 : (tensor<32x128x8x16xf32>, tensor<32x128x8x16xf32>) -> tensor<32x128x8x32xf32>
    %443 = stablehlo.transpose %442, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %444 = stablehlo.slice %427 [0:32, 0:128, 0:8, 0:16] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %445 = stablehlo.slice %427 [0:32, 0:128, 0:8, 16:32] : (tensor<32x128x8x32xf32>) -> tensor<32x128x8x16xf32>
    %446 = stablehlo.multiply %444, %434 : tensor<32x128x8x16xf32>
    %447 = stablehlo.multiply %445, %436 : tensor<32x128x8x16xf32>
    %448 = stablehlo.subtract %446, %447 : tensor<32x128x8x16xf32>
    %449 = stablehlo.multiply %445, %434 : tensor<32x128x8x16xf32>
    %450 = stablehlo.multiply %444, %436 : tensor<32x128x8x16xf32>
    %451 = stablehlo.add %449, %450 : tensor<32x128x8x16xf32>
    %452 = stablehlo.concatenate %448, %451, dim = 3 : (tensor<32x128x8x16xf32>, tensor<32x128x8x16xf32>) -> tensor<32x128x8x32xf32>
    %453 = stablehlo.transpose %452, dims = [0, 2, 1, 3] : (tensor<32x128x8x32xf32>) -> tensor<32x8x128x32xf32>
    %454 = stablehlo.transpose %453, dims = [0, 1, 3, 2] : (tensor<32x8x128x32xf32>) -> tensor<32x8x32x128xf32>
    %455 = stablehlo.reshape %443 : (tensor<32x8x128x32xf32>) -> tensor<256x128x32xf32>
    %456 = stablehlo.reshape %454 : (tensor<32x8x32x128xf32>) -> tensor<256x32x128xf32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2] : (tensor<256x32x128xf32>) -> tensor<256x32x128xf32>
    %458 = stablehlo.dot_general %455, %457, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x32xf32>, tensor<256x32x128xf32>) -> tensor<256x128x128xf32>
    %459 = stablehlo.reshape %458 : (tensor<256x128x128xf32>) -> tensor<32x8x128x128xf32>
    %460 = stablehlo.multiply %459, %78 : tensor<32x8x128x128xf32>
    %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1, 2, 3] : (tensor<32x8x128x128xf32>) -> tensor<32x8x128x128xf32>
    %462 = stablehlo.select %90, %91, %461 : tensor<32x8x128x128xi1>, tensor<32x8x128x128xf32>
    %463 = stablehlo.reduce(%462 init: %cst_3) applies stablehlo.maximum across dimensions = [3] : (tensor<32x8x128x128xf32>, tensor<f32>) -> tensor<32x8x128xf32>
    %464 = stablehlo.reshape %463 : (tensor<32x8x128xf32>) -> tensor<32x8x128x1xf32>
    %465 = stablehlo.broadcast_in_dim %464, dims = [0, 1, 2, 3] : (tensor<32x8x128x1xf32>) -> tensor<32x8x128x128xf32>
    %466 = stablehlo.subtract %462, %465 : tensor<32x8x128x128xf32>
    %467 = stablehlo.exponential %466 : tensor<32x8x128x128xf32>
    %468 = stablehlo.reduce(%467 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<32x8x128x128xf32>, tensor<f32>) -> tensor<32x8x128xf32>
    %469 = stablehlo.reshape %468 : (tensor<32x8x128xf32>) -> tensor<32x8x128x1xf32>
    %470 = stablehlo.broadcast_in_dim %469, dims = [0, 1, 2, 3] : (tensor<32x8x128x1xf32>) -> tensor<32x8x128x128xf32>
    %471 = stablehlo.divide %467, %470 : tensor<32x8x128x128xf32>
    %472 = stablehlo.reshape %471 : (tensor<32x8x128x128xf32>) -> tensor<256x128x128xf32>
    %473 = stablehlo.reshape %429 : (tensor<32x8x128x32xf32>) -> tensor<256x128x32xf32>
    %474 = stablehlo.broadcast_in_dim %473, dims = [0, 1, 2] : (tensor<256x128x32xf32>) -> tensor<256x128x32xf32>
    %475 = stablehlo.dot_general %472, %474, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x128x128xf32>, tensor<256x128x32xf32>) -> tensor<256x128x32xf32>
    %476 = stablehlo.reshape %475 : (tensor<256x128x32xf32>) -> tensor<32x8x128x32xf32>
    %477 = stablehlo.transpose %476, dims = [0, 2, 1, 3] : (tensor<32x8x128x32xf32>) -> tensor<32x128x8x32xf32>
    %478 = stablehlo.reshape %477 : (tensor<32x128x8x32xf32>) -> tensor<32x128x256xf32>
    %479 = stablehlo.transpose %arg36, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %480 = stablehlo.reshape %478 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %481 = stablehlo.dot_general %480, %479, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %482 = stablehlo.reshape %481 : (tensor<4096x256xf32>) -> tensor<32x128x256xf32>
    %483 = stablehlo.add %394, %482 : tensor<32x128x256xf32>
    %484 = stablehlo.convert %483 : (tensor<32x128x256xf32>) -> tensor<32x128x256xf64>
    %485 = stablehlo.reduce(%484 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %486 = stablehlo.reshape %485 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %487 = stablehlo.divide %486, %7 : tensor<32x128x1xf64>
    %488 = stablehlo.broadcast_in_dim %487, dims = [0, 1, 2] : (tensor<32x128x1xf64>) -> tensor<32x128x256xf64>
    %489 = stablehlo.subtract %484, %488 : tensor<32x128x256xf64>
    %490 = stablehlo.multiply %489, %489 : tensor<32x128x256xf64>
    %491 = stablehlo.reduce(%490 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %492 = stablehlo.reshape %491 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %493 = stablehlo.divide %492, %7 : tensor<32x128x1xf64>
    %494 = stablehlo.convert %493 : (tensor<32x128x1xf64>) -> tensor<32x128x1xf32>
    %495 = stablehlo.reduce(%483 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf32>, tensor<f32>) -> tensor<32x128xf32>
    %496 = stablehlo.reshape %495 : (tensor<32x128xf32>) -> tensor<32x128x1xf32>
    %497 = stablehlo.divide %496, %20 : tensor<32x128x1xf32>
    %498 = stablehlo.add %494, %24 : tensor<32x128x1xf32>
    %499 = stablehlo.rsqrt %498 : tensor<32x128x1xf32>
    %500 = stablehlo.broadcast_in_dim %497, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %501 = stablehlo.subtract %483, %500 : tensor<32x128x256xf32>
    %502 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %503 = stablehlo.multiply %501, %502 : tensor<32x128x256xf32>
    %504 = stablehlo.broadcast_in_dim %arg37, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %505 = stablehlo.multiply %503, %504 : tensor<32x128x256xf32>
    %506 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %507 = stablehlo.add %505, %506 : tensor<32x128x256xf32>
    %508 = stablehlo.transpose %arg39, dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %509 = stablehlo.reshape %507 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %510 = stablehlo.dot_general %509, %508, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x1024xf32>) -> tensor<4096x1024xf32>
    %511 = stablehlo.reshape %510 : (tensor<4096x1024xf32>) -> tensor<32x128x1024xf32>
    %512 = stablehlo.maximum %511, %cst_7 : tensor<32x128x1024xf32>
    %513 = stablehlo.transpose %arg40, dims = [1, 0] : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
    %514 = stablehlo.reshape %512 : (tensor<32x128x1024xf32>) -> tensor<4096x1024xf32>
    %515 = stablehlo.dot_general %514, %513, contracting_dims = [1] x [0] : (tensor<4096x1024xf32>, tensor<1024x256xf32>) -> tensor<4096x256xf32>
    %516 = stablehlo.reshape %515 : (tensor<4096x256xf32>) -> tensor<32x128x256xf32>
    %517 = stablehlo.add %483, %516 : tensor<32x128x256xf32>
    %518 = stablehlo.convert %517 : (tensor<32x128x256xf32>) -> tensor<32x128x256xf64>
    %519 = stablehlo.reduce(%518 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %520 = stablehlo.reshape %519 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %521 = stablehlo.divide %520, %7 : tensor<32x128x1xf64>
    %522 = stablehlo.broadcast_in_dim %521, dims = [0, 1, 2] : (tensor<32x128x1xf64>) -> tensor<32x128x256xf64>
    %523 = stablehlo.subtract %518, %522 : tensor<32x128x256xf64>
    %524 = stablehlo.multiply %523, %523 : tensor<32x128x256xf64>
    %525 = stablehlo.reduce(%524 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf64>, tensor<f64>) -> tensor<32x128xf64>
    %526 = stablehlo.reshape %525 : (tensor<32x128xf64>) -> tensor<32x128x1xf64>
    %527 = stablehlo.divide %526, %7 : tensor<32x128x1xf64>
    %528 = stablehlo.convert %527 : (tensor<32x128x1xf64>) -> tensor<32x128x1xf32>
    %529 = stablehlo.reduce(%517 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<32x128x256xf32>, tensor<f32>) -> tensor<32x128xf32>
    %530 = stablehlo.reshape %529 : (tensor<32x128xf32>) -> tensor<32x128x1xf32>
    %531 = stablehlo.divide %530, %20 : tensor<32x128x1xf32>
    %532 = stablehlo.add %528, %24 : tensor<32x128x1xf32>
    %533 = stablehlo.rsqrt %532 : tensor<32x128x1xf32>
    %534 = stablehlo.broadcast_in_dim %531, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %535 = stablehlo.subtract %517, %534 : tensor<32x128x256xf32>
    %536 = stablehlo.broadcast_in_dim %533, dims = [0, 1, 2] : (tensor<32x128x1xf32>) -> tensor<32x128x256xf32>
    %537 = stablehlo.multiply %535, %536 : tensor<32x128x256xf32>
    %538 = stablehlo.broadcast_in_dim %arg41, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %539 = stablehlo.multiply %537, %538 : tensor<32x128x256xf32>
    %540 = stablehlo.broadcast_in_dim %arg42, dims = [2] : (tensor<256xf32>) -> tensor<32x128x256xf32>
    %541 = stablehlo.add %539, %540 : tensor<32x128x256xf32>
    %542 = stablehlo.reshape %541 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %543 = stablehlo.transpose %arg43, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %544 = stablehlo.dot_general %542, %543, contracting_dims = [1] x [0] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %545 = stablehlo.convert %cst_11 : (tensor<1xi64>) -> tensor<1xf32>
    %546 = stablehlo.reshape %545 : (tensor<1xf32>) -> tensor<f32>
    %547 = stablehlo.broadcast_in_dim %546, dims = [] : (tensor<f32>) -> tensor<4096x256xf32>
    %548 = stablehlo.multiply %544, %547 : tensor<4096x256xf32>
    %549 = stablehlo.broadcast_in_dim %546, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %550 = stablehlo.multiply %arg44, %549 : tensor<256xf32>
    %551 = stablehlo.broadcast_in_dim %550, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
    %552 = stablehlo.add %548, %551 : tensor<4096x256xf32>
    %553 = stablehlo.reshape %552 : (tensor<4096x256xf32>) -> tensor<32x128x256xf32>
    %554 = stablehlo.reshape %553 : (tensor<32x128x256xf32>) -> tensor<4096x256xf32>
    %555 = stablehlo.reshape %arg48 : (tensor<32x128xi64>) -> tensor<4096xi64>
    %556 = stablehlo.reduce(%554 init: %cst_3) applies stablehlo.maximum across dimensions = [1] : (tensor<4096x256xf32>, tensor<f32>) -> tensor<4096xf32>
    %557 = stablehlo.reshape %556 : (tensor<4096xf32>) -> tensor<4096x1xf32>
    %558 = stablehlo.broadcast_in_dim %557, dims = [0, 1] : (tensor<4096x1xf32>) -> tensor<4096x256xf32>
    %559 = stablehlo.subtract %554, %558 : tensor<4096x256xf32>
    %560 = stablehlo.exponential %559 : tensor<4096x256xf32>
    %561 = stablehlo.reduce(%560 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4096x256xf32>, tensor<f32>) -> tensor<4096xf32>
    %562 = stablehlo.reshape %561 : (tensor<4096xf32>) -> tensor<4096x1xf32>
    %563 = stablehlo.log %562 : tensor<4096x1xf32>
    %564 = stablehlo.broadcast_in_dim %563, dims = [0, 1] : (tensor<4096x1xf32>) -> tensor<4096x256xf32>
    %565 = stablehlo.subtract %559, %564 : tensor<4096x256xf32>
    %566 = stablehlo.compare  NE, %555, %c_0,  SIGNED : (tensor<4096xi64>, tensor<4096xi64>) -> tensor<4096xi1>
    %567 = stablehlo.broadcast_in_dim %566, dims = [0] : (tensor<4096xi1>) -> tensor<4096xi1>
    %568 = stablehlo.broadcast_in_dim %555, dims = [0] : (tensor<4096xi64>) -> tensor<4096xi64>
    %569 = stablehlo.select %567, %568, %c : tensor<4096xi1>, tensor<4096xi64>
    %570 = stablehlo.reshape %569 : (tensor<4096xi64>) -> tensor<4096x1xi64>
    %571 = stablehlo.iota dim = 0 : tensor<4096x1x1xi64>
    %572 = stablehlo.reshape %570 : (tensor<4096x1xi64>) -> tensor<4096x1x1xi64>
    %573 = stablehlo.concatenate %571, %572, dim = 2 : (tensor<4096x1x1xi64>, tensor<4096x1x1xi64>) -> tensor<4096x1x2xi64>
    %574 = "stablehlo.gather"(%565, %573) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<4096x256xf32>, tensor<4096x1x2xi64>) -> tensor<4096x1xf32>
    %575 = stablehlo.reshape %574 : (tensor<4096x1xf32>) -> tensor<4096xf32>
    %576 = stablehlo.negate %575 : tensor<4096xf32>
    %577 = stablehlo.broadcast_in_dim %576, dims = [0] : (tensor<4096xf32>) -> tensor<4096xf32>
    %578 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4096xf32>
    %579 = stablehlo.select %567, %577, %578 : tensor<4096xi1>, tensor<4096xf32>
    %580 = stablehlo.convert %566 : (tensor<4096xi1>) -> tensor<4096xi64>
    %581 = stablehlo.reduce(%580 init: %c_2) applies stablehlo.add across dimensions = [0] : (tensor<4096xi64>, tensor<i64>) -> tensor<i64>
    %582 = stablehlo.convert %581 : (tensor<i64>) -> tensor<f32>
    %583 = stablehlo.reduce(%579 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4096xf32>, tensor<f32>) -> tensor<f32>
    %584 = stablehlo.divide %583, %582 : tensor<f32>
    return %584 : tensor<f32>
  }
}
