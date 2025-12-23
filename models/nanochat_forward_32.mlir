module {
  func.func @main(%arg0: tensor<1x32x1x8xbf16>, %arg1: tensor<1x32x1x8xbf16>, %arg2: tensor<65x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>, %arg5: tensor<64x64xf32>, %arg6: tensor<64x64xf32>, %arg7: tensor<256x64xf32>, %arg8: tensor<64x256xf32>, %arg9: tensor<64x64xf32>, %arg10: tensor<64x64xf32>, %arg11: tensor<64x64xf32>, %arg12: tensor<64x64xf32>, %arg13: tensor<256x64xf32>, %arg14: tensor<64x256xf32>, %arg15: tensor<65x64xf32>, %arg16: tensor<64x32xi64>, %arg17: tensor<64x32xi64>) -> tensor<f32> {
    %c = stablehlo.constant dense<0> : tensor<2048xi64>
    %c_0 = stablehlo.constant dense<-1> : tensor<2048xi64>
    %c_1 = stablehlo.constant dense<1> : tensor<32x32xi64>
    %c_2 = stablehlo.constant dense<0> : tensor<32xi64>
    %c_3 = stablehlo.constant dense<1> : tensor<32xi64>
    %c_4 = stablehlo.constant dense<32> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %cst_7 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_8 = stablehlo.constant dense<true> : tensor<32x32xi1>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<64x32x256xf32>
    %cst_10 = arith.constant dense<2> : tensor<1xi64>
    %cst_11 = arith.constant dense<64> : tensor<1xi64>
    %cst_12 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_13 = arith.constant dense<16> : tensor<1xi64>
    %cst_14 = arith.constant dense<2.500000e-01> : tensor<1xf64>
    %cst_15 = arith.constant dense<15> : tensor<1xi64>
    %0 = "stablehlo.gather"(%arg2, %arg16) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<65x64xf32>, tensor<64x32xi64>) -> tensor<64x32x64xf32>
    %1 = stablehlo.convert %0 : tensor<64x32x64xf32>
    %2 = stablehlo.convert %cst_10 : (tensor<1xi64>) -> tensor<1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<64x32x64xf32>
    %5 = stablehlo.power %1, %4 : tensor<64x32x64xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %7 = stablehlo.reshape %6 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %8 = stablehlo.convert %cst_11 : (tensor<1xi64>) -> tensor<1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<64x32x1xf32>
    %11 = stablehlo.divide %7, %10 : tensor<64x32x1xf32>
    %12 = stablehlo.convert %cst_12 : (tensor<1xf64>) -> tensor<1xf32>
    %13 = stablehlo.reshape %12 : (tensor<1xf32>) -> tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<64x32x1xf32>
    %15 = stablehlo.add %11, %14 : tensor<64x32x1xf32>
    %16 = stablehlo.rsqrt %15 : tensor<64x32x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %18 = stablehlo.multiply %1, %17 : tensor<64x32x64xf32>
    %19 = stablehlo.power %18, %4 : tensor<64x32x64xf32>
    %20 = stablehlo.reduce(%19 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %21 = stablehlo.reshape %20 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %22 = stablehlo.divide %21, %10 : tensor<64x32x1xf32>
    %23 = stablehlo.add %22, %14 : tensor<64x32x1xf32>
    %24 = stablehlo.rsqrt %23 : tensor<64x32x1xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %26 = stablehlo.multiply %18, %25 : tensor<64x32x64xf32>
    %27 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %28 = stablehlo.reshape %26 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %29 = stablehlo.dot_general %28, %27, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %30 = stablehlo.reshape %29 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %31 = stablehlo.reshape %30 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %32 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %33 = stablehlo.dot_general %28, %32, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %34 = stablehlo.reshape %33 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %35 = stablehlo.reshape %34 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %36 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %37 = stablehlo.dot_general %28, %36, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %38 = stablehlo.reshape %37 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %39 = stablehlo.reshape %38 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %40 = stablehlo.slice %31 [0:64, 0:32, 0:4, 0:8] : (tensor<64x32x4x16xf32>) -> tensor<64x32x4x8xf32>
    %41 = stablehlo.slice %31 [0:64, 0:32, 0:4, 8:16] : (tensor<64x32x4x16xf32>) -> tensor<64x32x4x8xf32>
    %42 = stablehlo.convert %arg0 : (tensor<1x32x1x8xbf16>) -> tensor<1x32x1x8xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2, 3] : (tensor<1x32x1x8xf32>) -> tensor<64x32x4x8xf32>
    %44 = stablehlo.multiply %40, %43 : tensor<64x32x4x8xf32>
    %45 = stablehlo.convert %arg1 : (tensor<1x32x1x8xbf16>) -> tensor<1x32x1x8xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x32x1x8xf32>) -> tensor<64x32x4x8xf32>
    %47 = stablehlo.multiply %41, %46 : tensor<64x32x4x8xf32>
    %48 = stablehlo.add %44, %47 : tensor<64x32x4x8xf32>
    %49 = stablehlo.negate %arg1 : tensor<1x32x1x8xbf16>
    %50 = stablehlo.convert %49 : (tensor<1x32x1x8xbf16>) -> tensor<1x32x1x8xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2, 3] : (tensor<1x32x1x8xf32>) -> tensor<64x32x4x8xf32>
    %52 = stablehlo.multiply %40, %51 : tensor<64x32x4x8xf32>
    %53 = stablehlo.multiply %41, %43 : tensor<64x32x4x8xf32>
    %54 = stablehlo.add %52, %53 : tensor<64x32x4x8xf32>
    %55 = stablehlo.concatenate %48, %54, dim = 3 : (tensor<64x32x4x8xf32>, tensor<64x32x4x8xf32>) -> tensor<64x32x4x16xf32>
    %56 = stablehlo.slice %35 [0:64, 0:32, 0:4, 0:8] : (tensor<64x32x4x16xf32>) -> tensor<64x32x4x8xf32>
    %57 = stablehlo.slice %35 [0:64, 0:32, 0:4, 8:16] : (tensor<64x32x4x16xf32>) -> tensor<64x32x4x8xf32>
    %58 = stablehlo.multiply %56, %43 : tensor<64x32x4x8xf32>
    %59 = stablehlo.multiply %57, %46 : tensor<64x32x4x8xf32>
    %60 = stablehlo.add %58, %59 : tensor<64x32x4x8xf32>
    %61 = stablehlo.multiply %56, %51 : tensor<64x32x4x8xf32>
    %62 = stablehlo.multiply %57, %43 : tensor<64x32x4x8xf32>
    %63 = stablehlo.add %61, %62 : tensor<64x32x4x8xf32>
    %64 = stablehlo.concatenate %60, %63, dim = 3 : (tensor<64x32x4x8xf32>, tensor<64x32x4x8xf32>) -> tensor<64x32x4x16xf32>
    %65 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<64x32x4x16xf32>
    %66 = stablehlo.power %55, %65 : tensor<64x32x4x16xf32>
    %67 = stablehlo.reduce(%66 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x32x4x16xf32>, tensor<f32>) -> tensor<64x32x4xf32>
    %68 = stablehlo.reshape %67 : (tensor<64x32x4xf32>) -> tensor<64x32x4x1xf32>
    %69 = stablehlo.convert %cst_13 : (tensor<1xi64>) -> tensor<1xf32>
    %70 = stablehlo.reshape %69 : (tensor<1xf32>) -> tensor<f32>
    %71 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<f32>) -> tensor<64x32x4x1xf32>
    %72 = stablehlo.divide %68, %71 : tensor<64x32x4x1xf32>
    %73 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<64x32x4x1xf32>
    %74 = stablehlo.add %72, %73 : tensor<64x32x4x1xf32>
    %75 = stablehlo.rsqrt %74 : tensor<64x32x4x1xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<64x32x4x1xf32>) -> tensor<64x32x4x16xf32>
    %77 = stablehlo.multiply %55, %76 : tensor<64x32x4x16xf32>
    %78 = stablehlo.power %64, %65 : tensor<64x32x4x16xf32>
    %79 = stablehlo.reduce(%78 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x32x4x16xf32>, tensor<f32>) -> tensor<64x32x4xf32>
    %80 = stablehlo.reshape %79 : (tensor<64x32x4xf32>) -> tensor<64x32x4x1xf32>
    %81 = stablehlo.divide %80, %71 : tensor<64x32x4x1xf32>
    %82 = stablehlo.add %81, %73 : tensor<64x32x4x1xf32>
    %83 = stablehlo.rsqrt %82 : tensor<64x32x4x1xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<64x32x4x1xf32>) -> tensor<64x32x4x16xf32>
    %85 = stablehlo.multiply %64, %84 : tensor<64x32x4x16xf32>
    %86 = stablehlo.transpose %77, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %87 = stablehlo.transpose %85, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %88 = stablehlo.transpose %39, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %89 = stablehlo.transpose %87, dims = [0, 1, 3, 2] : (tensor<64x4x32x16xf32>) -> tensor<64x4x16x32xf32>
    %90 = stablehlo.reshape %86 : (tensor<64x4x32x16xf32>) -> tensor<256x32x16xf32>
    %91 = stablehlo.reshape %89 : (tensor<64x4x16x32xf32>) -> tensor<256x16x32xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2] : (tensor<256x16x32xf32>) -> tensor<256x16x32xf32>
    %93 = stablehlo.dot_general %90, %92, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x32x16xf32>, tensor<256x16x32xf32>) -> tensor<256x32x32xf32>
    %94 = stablehlo.reshape %93 : (tensor<256x32x32xf32>) -> tensor<64x4x32x32xf32>
    %95 = stablehlo.convert %cst_14 : (tensor<1xf64>) -> tensor<1xf32>
    %96 = stablehlo.reshape %95 : (tensor<1xf32>) -> tensor<f32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<f32>) -> tensor<64x4x32x32xf32>
    %98 = stablehlo.multiply %94, %97 : tensor<64x4x32x32xf32>
    %99 = stablehlo.convert %c_5 : (tensor<i64>) -> tensor<f64>
    %100 = stablehlo.convert %c_4 : (tensor<i64>) -> tensor<f64>
    %101 = stablehlo.divide %100, %99 : tensor<f64>
    %102 = stablehlo.ceil %101 : tensor<f64>
    %103 = stablehlo.convert %102 : (tensor<f64>) -> tensor<i64>
    %104 = stablehlo.reshape %103 : (tensor<i64>) -> tensor<1xi64>
    %105 = stablehlo.dynamic_iota %104, dim = 0 : (tensor<1xi64>) -> tensor<32xi64>
    %106 = stablehlo.multiply %105, %c_3 : tensor<32xi64>
    %107 = stablehlo.add %106, %c_2 : tensor<32xi64>
    %108 = stablehlo.reshape %107 : (tensor<32xi64>) -> tensor<1x32xi64>
    %109 = stablehlo.reshape %107 : (tensor<32xi64>) -> tensor<32x1xi64>
    %110 = stablehlo.broadcast_in_dim %108, dims = [0, 1] : (tensor<1x32xi64>) -> tensor<32x32xi64>
    %111 = stablehlo.broadcast_in_dim %109, dims = [0, 1] : (tensor<32x1xi64>) -> tensor<32x32xi64>
    %112 = stablehlo.subtract %110, %111 : tensor<32x32xi64>
    %113 = stablehlo.compare  GE, %112, %c_1,  SIGNED : (tensor<32x32xi64>, tensor<32x32xi64>) -> tensor<32x32xi1>
    %114 = stablehlo.and %113, %c_8 : tensor<32x32xi1>
    %115 = stablehlo.reshape %114 : (tensor<32x32xi1>) -> tensor<1x1x32x32xi1>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0, 1, 2, 3] : (tensor<1x1x32x32xi1>) -> tensor<64x4x32x32xi1>
    %117 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x4x32x32xf32>
    %118 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2, 3] : (tensor<64x4x32x32xf32>) -> tensor<64x4x32x32xf32>
    %119 = stablehlo.select %116, %117, %118 : tensor<64x4x32x32xi1>, tensor<64x4x32x32xf32>
    %120 = stablehlo.reduce(%119 init: %cst_7) applies stablehlo.maximum across dimensions = [3] : (tensor<64x4x32x32xf32>, tensor<f32>) -> tensor<64x4x32xf32>
    %121 = stablehlo.reshape %120 : (tensor<64x4x32xf32>) -> tensor<64x4x32x1xf32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2, 3] : (tensor<64x4x32x1xf32>) -> tensor<64x4x32x32xf32>
    %123 = stablehlo.subtract %119, %122 : tensor<64x4x32x32xf32>
    %124 = stablehlo.exponential %123 : tensor<64x4x32x32xf32>
    %125 = stablehlo.reduce(%124 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x4x32x32xf32>, tensor<f32>) -> tensor<64x4x32xf32>
    %126 = stablehlo.reshape %125 : (tensor<64x4x32xf32>) -> tensor<64x4x32x1xf32>
    %127 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2, 3] : (tensor<64x4x32x1xf32>) -> tensor<64x4x32x32xf32>
    %128 = stablehlo.divide %124, %127 : tensor<64x4x32x32xf32>
    %129 = stablehlo.reshape %128 : (tensor<64x4x32x32xf32>) -> tensor<256x32x32xf32>
    %130 = stablehlo.reshape %88 : (tensor<64x4x32x16xf32>) -> tensor<256x32x16xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<256x32x16xf32>) -> tensor<256x32x16xf32>
    %132 = stablehlo.dot_general %129, %131, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x32x32xf32>, tensor<256x32x16xf32>) -> tensor<256x32x16xf32>
    %133 = stablehlo.reshape %132 : (tensor<256x32x16xf32>) -> tensor<64x4x32x16xf32>
    %134 = stablehlo.transpose %133, dims = [0, 2, 1, 3] : (tensor<64x4x32x16xf32>) -> tensor<64x32x4x16xf32>
    %135 = stablehlo.reshape %134 : (tensor<64x32x4x16xf32>) -> tensor<64x32x64xf32>
    %136 = stablehlo.transpose %arg6, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %137 = stablehlo.reshape %135 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %138 = stablehlo.dot_general %137, %136, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %139 = stablehlo.reshape %138 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %140 = stablehlo.add %18, %139 : tensor<64x32x64xf32>
    %141 = stablehlo.power %140, %4 : tensor<64x32x64xf32>
    %142 = stablehlo.reduce(%141 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %143 = stablehlo.reshape %142 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %144 = stablehlo.divide %143, %10 : tensor<64x32x1xf32>
    %145 = stablehlo.add %144, %14 : tensor<64x32x1xf32>
    %146 = stablehlo.rsqrt %145 : tensor<64x32x1xf32>
    %147 = stablehlo.broadcast_in_dim %146, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %148 = stablehlo.multiply %140, %147 : tensor<64x32x64xf32>
    %149 = stablehlo.transpose %arg7, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
    %150 = stablehlo.reshape %148 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %151 = stablehlo.dot_general %150, %149, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x256xf32>) -> tensor<2048x256xf32>
    %152 = stablehlo.reshape %151 : (tensor<2048x256xf32>) -> tensor<64x32x256xf32>
    %153 = stablehlo.maximum %152, %cst_9 : tensor<64x32x256xf32>
    %154 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<64x32x256xf32>
    %155 = stablehlo.power %153, %154 : tensor<64x32x256xf32>
    %156 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<64x256xf32>) -> tensor<256x64xf32>
    %157 = stablehlo.reshape %155 : (tensor<64x32x256xf32>) -> tensor<2048x256xf32>
    %158 = stablehlo.dot_general %157, %156, contracting_dims = [1] x [0] : (tensor<2048x256xf32>, tensor<256x64xf32>) -> tensor<2048x64xf32>
    %159 = stablehlo.reshape %158 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %160 = stablehlo.add %140, %159 : tensor<64x32x64xf32>
    %161 = stablehlo.power %160, %4 : tensor<64x32x64xf32>
    %162 = stablehlo.reduce(%161 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %163 = stablehlo.reshape %162 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %164 = stablehlo.divide %163, %10 : tensor<64x32x1xf32>
    %165 = stablehlo.add %164, %14 : tensor<64x32x1xf32>
    %166 = stablehlo.rsqrt %165 : tensor<64x32x1xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %168 = stablehlo.multiply %160, %167 : tensor<64x32x64xf32>
    %169 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %170 = stablehlo.reshape %168 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %171 = stablehlo.dot_general %170, %169, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %172 = stablehlo.reshape %171 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %173 = stablehlo.reshape %172 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %174 = stablehlo.transpose %arg10, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %175 = stablehlo.dot_general %170, %174, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %176 = stablehlo.reshape %175 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %177 = stablehlo.reshape %176 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %178 = stablehlo.transpose %arg11, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %179 = stablehlo.dot_general %170, %178, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %180 = stablehlo.reshape %179 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %181 = stablehlo.reshape %180 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %182 = stablehlo.slice %173 [0:64, 0:32, 0:4, 0:8] : (tensor<64x32x4x16xf32>) -> tensor<64x32x4x8xf32>
    %183 = stablehlo.slice %173 [0:64, 0:32, 0:4, 8:16] : (tensor<64x32x4x16xf32>) -> tensor<64x32x4x8xf32>
    %184 = stablehlo.multiply %182, %43 : tensor<64x32x4x8xf32>
    %185 = stablehlo.multiply %183, %46 : tensor<64x32x4x8xf32>
    %186 = stablehlo.add %184, %185 : tensor<64x32x4x8xf32>
    %187 = stablehlo.multiply %182, %51 : tensor<64x32x4x8xf32>
    %188 = stablehlo.multiply %183, %43 : tensor<64x32x4x8xf32>
    %189 = stablehlo.add %187, %188 : tensor<64x32x4x8xf32>
    %190 = stablehlo.concatenate %186, %189, dim = 3 : (tensor<64x32x4x8xf32>, tensor<64x32x4x8xf32>) -> tensor<64x32x4x16xf32>
    %191 = stablehlo.slice %177 [0:64, 0:32, 0:4, 0:8] : (tensor<64x32x4x16xf32>) -> tensor<64x32x4x8xf32>
    %192 = stablehlo.slice %177 [0:64, 0:32, 0:4, 8:16] : (tensor<64x32x4x16xf32>) -> tensor<64x32x4x8xf32>
    %193 = stablehlo.multiply %191, %43 : tensor<64x32x4x8xf32>
    %194 = stablehlo.multiply %192, %46 : tensor<64x32x4x8xf32>
    %195 = stablehlo.add %193, %194 : tensor<64x32x4x8xf32>
    %196 = stablehlo.multiply %191, %51 : tensor<64x32x4x8xf32>
    %197 = stablehlo.multiply %192, %43 : tensor<64x32x4x8xf32>
    %198 = stablehlo.add %196, %197 : tensor<64x32x4x8xf32>
    %199 = stablehlo.concatenate %195, %198, dim = 3 : (tensor<64x32x4x8xf32>, tensor<64x32x4x8xf32>) -> tensor<64x32x4x16xf32>
    %200 = stablehlo.power %190, %65 : tensor<64x32x4x16xf32>
    %201 = stablehlo.reduce(%200 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x32x4x16xf32>, tensor<f32>) -> tensor<64x32x4xf32>
    %202 = stablehlo.reshape %201 : (tensor<64x32x4xf32>) -> tensor<64x32x4x1xf32>
    %203 = stablehlo.divide %202, %71 : tensor<64x32x4x1xf32>
    %204 = stablehlo.add %203, %73 : tensor<64x32x4x1xf32>
    %205 = stablehlo.rsqrt %204 : tensor<64x32x4x1xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<64x32x4x1xf32>) -> tensor<64x32x4x16xf32>
    %207 = stablehlo.multiply %190, %206 : tensor<64x32x4x16xf32>
    %208 = stablehlo.power %199, %65 : tensor<64x32x4x16xf32>
    %209 = stablehlo.reduce(%208 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x32x4x16xf32>, tensor<f32>) -> tensor<64x32x4xf32>
    %210 = stablehlo.reshape %209 : (tensor<64x32x4xf32>) -> tensor<64x32x4x1xf32>
    %211 = stablehlo.divide %210, %71 : tensor<64x32x4x1xf32>
    %212 = stablehlo.add %211, %73 : tensor<64x32x4x1xf32>
    %213 = stablehlo.rsqrt %212 : tensor<64x32x4x1xf32>
    %214 = stablehlo.broadcast_in_dim %213, dims = [0, 1, 2, 3] : (tensor<64x32x4x1xf32>) -> tensor<64x32x4x16xf32>
    %215 = stablehlo.multiply %199, %214 : tensor<64x32x4x16xf32>
    %216 = stablehlo.transpose %207, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %217 = stablehlo.transpose %215, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %218 = stablehlo.transpose %181, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %219 = stablehlo.transpose %217, dims = [0, 1, 3, 2] : (tensor<64x4x32x16xf32>) -> tensor<64x4x16x32xf32>
    %220 = stablehlo.reshape %216 : (tensor<64x4x32x16xf32>) -> tensor<256x32x16xf32>
    %221 = stablehlo.reshape %219 : (tensor<64x4x16x32xf32>) -> tensor<256x16x32xf32>
    %222 = stablehlo.broadcast_in_dim %221, dims = [0, 1, 2] : (tensor<256x16x32xf32>) -> tensor<256x16x32xf32>
    %223 = stablehlo.dot_general %220, %222, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x32x16xf32>, tensor<256x16x32xf32>) -> tensor<256x32x32xf32>
    %224 = stablehlo.reshape %223 : (tensor<256x32x32xf32>) -> tensor<64x4x32x32xf32>
    %225 = stablehlo.multiply %224, %97 : tensor<64x4x32x32xf32>
    %226 = stablehlo.broadcast_in_dim %225, dims = [0, 1, 2, 3] : (tensor<64x4x32x32xf32>) -> tensor<64x4x32x32xf32>
    %227 = stablehlo.select %116, %117, %226 : tensor<64x4x32x32xi1>, tensor<64x4x32x32xf32>
    %228 = stablehlo.reduce(%227 init: %cst_7) applies stablehlo.maximum across dimensions = [3] : (tensor<64x4x32x32xf32>, tensor<f32>) -> tensor<64x4x32xf32>
    %229 = stablehlo.reshape %228 : (tensor<64x4x32xf32>) -> tensor<64x4x32x1xf32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2, 3] : (tensor<64x4x32x1xf32>) -> tensor<64x4x32x32xf32>
    %231 = stablehlo.subtract %227, %230 : tensor<64x4x32x32xf32>
    %232 = stablehlo.exponential %231 : tensor<64x4x32x32xf32>
    %233 = stablehlo.reduce(%232 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x4x32x32xf32>, tensor<f32>) -> tensor<64x4x32xf32>
    %234 = stablehlo.reshape %233 : (tensor<64x4x32xf32>) -> tensor<64x4x32x1xf32>
    %235 = stablehlo.broadcast_in_dim %234, dims = [0, 1, 2, 3] : (tensor<64x4x32x1xf32>) -> tensor<64x4x32x32xf32>
    %236 = stablehlo.divide %232, %235 : tensor<64x4x32x32xf32>
    %237 = stablehlo.reshape %236 : (tensor<64x4x32x32xf32>) -> tensor<256x32x32xf32>
    %238 = stablehlo.reshape %218 : (tensor<64x4x32x16xf32>) -> tensor<256x32x16xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2] : (tensor<256x32x16xf32>) -> tensor<256x32x16xf32>
    %240 = stablehlo.dot_general %237, %239, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x32x32xf32>, tensor<256x32x16xf32>) -> tensor<256x32x16xf32>
    %241 = stablehlo.reshape %240 : (tensor<256x32x16xf32>) -> tensor<64x4x32x16xf32>
    %242 = stablehlo.transpose %241, dims = [0, 2, 1, 3] : (tensor<64x4x32x16xf32>) -> tensor<64x32x4x16xf32>
    %243 = stablehlo.reshape %242 : (tensor<64x32x4x16xf32>) -> tensor<64x32x64xf32>
    %244 = stablehlo.transpose %arg12, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %245 = stablehlo.reshape %243 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %246 = stablehlo.dot_general %245, %244, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %247 = stablehlo.reshape %246 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %248 = stablehlo.add %160, %247 : tensor<64x32x64xf32>
    %249 = stablehlo.power %248, %4 : tensor<64x32x64xf32>
    %250 = stablehlo.reduce(%249 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %251 = stablehlo.reshape %250 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %252 = stablehlo.divide %251, %10 : tensor<64x32x1xf32>
    %253 = stablehlo.add %252, %14 : tensor<64x32x1xf32>
    %254 = stablehlo.rsqrt %253 : tensor<64x32x1xf32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %256 = stablehlo.multiply %248, %255 : tensor<64x32x64xf32>
    %257 = stablehlo.transpose %arg13, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
    %258 = stablehlo.reshape %256 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %259 = stablehlo.dot_general %258, %257, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x256xf32>) -> tensor<2048x256xf32>
    %260 = stablehlo.reshape %259 : (tensor<2048x256xf32>) -> tensor<64x32x256xf32>
    %261 = stablehlo.maximum %260, %cst_9 : tensor<64x32x256xf32>
    %262 = stablehlo.power %261, %154 : tensor<64x32x256xf32>
    %263 = stablehlo.transpose %arg14, dims = [1, 0] : (tensor<64x256xf32>) -> tensor<256x64xf32>
    %264 = stablehlo.reshape %262 : (tensor<64x32x256xf32>) -> tensor<2048x256xf32>
    %265 = stablehlo.dot_general %264, %263, contracting_dims = [1] x [0] : (tensor<2048x256xf32>, tensor<256x64xf32>) -> tensor<2048x64xf32>
    %266 = stablehlo.reshape %265 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %267 = stablehlo.add %248, %266 : tensor<64x32x64xf32>
    %268 = stablehlo.power %267, %4 : tensor<64x32x64xf32>
    %269 = stablehlo.reduce(%268 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %270 = stablehlo.reshape %269 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %271 = stablehlo.divide %270, %10 : tensor<64x32x1xf32>
    %272 = stablehlo.add %271, %14 : tensor<64x32x1xf32>
    %273 = stablehlo.rsqrt %272 : tensor<64x32x1xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %275 = stablehlo.multiply %267, %274 : tensor<64x32x64xf32>
    %276 = stablehlo.transpose %arg15, dims = [1, 0] : (tensor<65x64xf32>) -> tensor<64x65xf32>
    %277 = stablehlo.reshape %275 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %278 = stablehlo.dot_general %277, %276, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x65xf32>) -> tensor<2048x65xf32>
    %279 = stablehlo.reshape %278 : (tensor<2048x65xf32>) -> tensor<64x32x65xf32>
    %280 = stablehlo.convert %cst_15 : (tensor<1xi64>) -> tensor<1xf32>
    %281 = stablehlo.reshape %280 : (tensor<1xf32>) -> tensor<f32>
    %282 = stablehlo.broadcast_in_dim %281, dims = [] : (tensor<f32>) -> tensor<64x32x65xf32>
    %283 = stablehlo.divide %279, %282 : tensor<64x32x65xf32>
    %284 = stablehlo.tanh %283 : tensor<64x32x65xf32>
    %285 = stablehlo.multiply %284, %282 : tensor<64x32x65xf32>
    %286 = stablehlo.reshape %285 : (tensor<64x32x65xf32>) -> tensor<2048x65xf32>
    %287 = stablehlo.reshape %arg17 : (tensor<64x32xi64>) -> tensor<2048xi64>
    %288 = stablehlo.reduce(%286 init: %cst_7) applies stablehlo.maximum across dimensions = [1] : (tensor<2048x65xf32>, tensor<f32>) -> tensor<2048xf32>
    %289 = stablehlo.reshape %288 : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x65xf32>
    %291 = stablehlo.subtract %286, %290 : tensor<2048x65xf32>
    %292 = stablehlo.exponential %291 : tensor<2048x65xf32>
    %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2048x65xf32>, tensor<f32>) -> tensor<2048xf32>
    %294 = stablehlo.reshape %293 : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %295 = stablehlo.log %294 : tensor<2048x1xf32>
    %296 = stablehlo.broadcast_in_dim %295, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x65xf32>
    %297 = stablehlo.subtract %291, %296 : tensor<2048x65xf32>
    %298 = stablehlo.compare  NE, %287, %c_0,  SIGNED : (tensor<2048xi64>, tensor<2048xi64>) -> tensor<2048xi1>
    %299 = stablehlo.broadcast_in_dim %298, dims = [0] : (tensor<2048xi1>) -> tensor<2048xi1>
    %300 = stablehlo.broadcast_in_dim %287, dims = [0] : (tensor<2048xi64>) -> tensor<2048xi64>
    %301 = stablehlo.select %299, %300, %c : tensor<2048xi1>, tensor<2048xi64>
    %302 = stablehlo.reshape %301 : (tensor<2048xi64>) -> tensor<2048x1xi64>
    %303 = stablehlo.iota dim = 0 : tensor<2048x1x1xi64>
    %304 = stablehlo.reshape %302 : (tensor<2048x1xi64>) -> tensor<2048x1x1xi64>
    %305 = stablehlo.concatenate %303, %304, dim = 2 : (tensor<2048x1x1xi64>, tensor<2048x1x1xi64>) -> tensor<2048x1x2xi64>
    %306 = "stablehlo.gather"(%297, %305) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<2048x65xf32>, tensor<2048x1x2xi64>) -> tensor<2048x1xf32>
    %307 = stablehlo.reshape %306 : (tensor<2048x1xf32>) -> tensor<2048xf32>
    %308 = stablehlo.negate %307 : tensor<2048xf32>
    %309 = stablehlo.broadcast_in_dim %308, dims = [0] : (tensor<2048xf32>) -> tensor<2048xf32>
    %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %311 = stablehlo.select %299, %309, %310 : tensor<2048xi1>, tensor<2048xf32>
    %312 = stablehlo.convert %298 : (tensor<2048xi1>) -> tensor<2048xi64>
    %313 = stablehlo.reduce(%312 init: %c_6) applies stablehlo.add across dimensions = [0] : (tensor<2048xi64>, tensor<i64>) -> tensor<i64>
    %314 = stablehlo.convert %313 : (tensor<i64>) -> tensor<f32>
    %315 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2048xf32>, tensor<f32>) -> tensor<f32>
    %316 = stablehlo.divide %315, %314 : tensor<f32>
    return %316 : tensor<f32>
  }
}
