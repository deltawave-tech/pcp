module {
  func.func @main(%arg0: tensor<65x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<192x64xf32>, %arg5: tensor<64x64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<256x64xf32>, %arg9: tensor<64x256xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<192x64xf32>, %arg13: tensor<64x64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<256x64xf32>, %arg17: tensor<64x256xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<65x64xf32>, %arg21: tensor<65xf32>, %arg22: tensor<64x32xi64>, %arg23: tensor<64x32xi64>) -> tensor<f32> {
    %c = stablehlo.constant dense<0> : tensor<2048xi64>
    %c_0 = stablehlo.constant dense<-100> : tensor<2048xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<32x32xi64>
    %c_2 = stablehlo.constant dense<0> : tensor<32xi64>
    %c_3 = stablehlo.constant dense<1> : tensor<32xi64>
    %c_4 = stablehlo.constant dense<32> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %cst_7 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<32x32xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<64x32x256xf32>
    %cst_12 = arith.constant dense<1> : tensor<1xi64>
    %cst_13 = arith.constant dense<64> : tensor<1xi64>
    %cst_14 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_15 = arith.constant dense<2.500000e-01> : tensor<1xf64>
    %0 = "stablehlo.gather"(%arg0, %arg22) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<65x64xf32>, tensor<64x32xi64>) -> tensor<64x32x64xf32>
    %1 = stablehlo.convert %0 : tensor<64x32x64xf32>
    %2 = stablehlo.convert %c_5 : (tensor<i64>) -> tensor<f64>
    %3 = stablehlo.convert %c_4 : (tensor<i64>) -> tensor<f64>
    %4 = stablehlo.divide %3, %2 : tensor<f64>
    %5 = stablehlo.ceil %4 : tensor<f64>
    %6 = stablehlo.convert %5 : (tensor<f64>) -> tensor<i64>
    %7 = stablehlo.reshape %6 : (tensor<i64>) -> tensor<1xi64>
    %8 = stablehlo.dynamic_iota %7, dim = 0 : (tensor<1xi64>) -> tensor<32xi64>
    %9 = stablehlo.multiply %8, %c_3 : tensor<32xi64>
    %10 = stablehlo.add %9, %c_2 : tensor<32xi64>
    %11 = "stablehlo.gather"(%arg1, %10) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<32x64xf32>, tensor<32xi64>) -> tensor<32x64xf32>
    %12 = stablehlo.convert %11 : tensor<32x64xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [1, 2] : (tensor<32x64xf32>) -> tensor<64x32x64xf32>
    %14 = stablehlo.add %1, %13 : tensor<64x32x64xf32>
    %15 = stablehlo.convert %14 : (tensor<64x32x64xf32>) -> tensor<64x32x64xf64>
    %16 = stablehlo.reduce(%15 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf64>, tensor<f64>) -> tensor<64x32xf64>
    %17 = stablehlo.reshape %16 : (tensor<64x32xf64>) -> tensor<64x32x1xf64>
    %18 = stablehlo.convert %cst_13 : (tensor<1xi64>) -> tensor<1xf64>
    %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f64>) -> tensor<64x32x1xf64>
    %21 = stablehlo.divide %17, %20 : tensor<64x32x1xf64>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<64x32x1xf64>) -> tensor<64x32x64xf64>
    %23 = stablehlo.subtract %15, %22 : tensor<64x32x64xf64>
    %24 = stablehlo.multiply %23, %23 : tensor<64x32x64xf64>
    %25 = stablehlo.reduce(%24 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf64>, tensor<f64>) -> tensor<64x32xf64>
    %26 = stablehlo.reshape %25 : (tensor<64x32xf64>) -> tensor<64x32x1xf64>
    %27 = stablehlo.divide %26, %20 : tensor<64x32x1xf64>
    %28 = stablehlo.convert %27 : (tensor<64x32x1xf64>) -> tensor<64x32x1xf32>
    %29 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %30 = stablehlo.reshape %29 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %31 = stablehlo.convert %cst_13 : (tensor<1xi64>) -> tensor<1xf32>
    %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<64x32x1xf32>
    %34 = stablehlo.divide %30, %33 : tensor<64x32x1xf32>
    %35 = stablehlo.convert %cst_14 : (tensor<1xf64>) -> tensor<1xf32>
    %36 = stablehlo.reshape %35 : (tensor<1xf32>) -> tensor<f32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<f32>) -> tensor<64x32x1xf32>
    %38 = stablehlo.add %28, %37 : tensor<64x32x1xf32>
    %39 = stablehlo.rsqrt %38 : tensor<64x32x1xf32>
    %40 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %41 = stablehlo.subtract %14, %40 : tensor<64x32x64xf32>
    %42 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<64x32x64xf32>
    %44 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<64x32x64xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<64x32x64xf32>
    %46 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<64xf32>) -> tensor<64x32x64xf32>
    %47 = stablehlo.add %45, %46 : tensor<64x32x64xf32>
    %48 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<192x64xf32>) -> tensor<64x192xf32>
    %49 = stablehlo.reshape %47 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %50 = stablehlo.dot_general %49, %48, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x192xf32>) -> tensor<2048x192xf32>
    %51 = stablehlo.reshape %50 : (tensor<2048x192xf32>) -> tensor<64x32x192xf32>
    %52 = stablehlo.slice %51 [0:64, 0:32, 0:64] : (tensor<64x32x192xf32>) -> tensor<64x32x64xf32>
    %53 = stablehlo.slice %51 [0:64, 0:32, 64:128] : (tensor<64x32x192xf32>) -> tensor<64x32x64xf32>
    %54 = stablehlo.slice %51 [0:64, 0:32, 128:192] : (tensor<64x32x192xf32>) -> tensor<64x32x64xf32>
    %55 = stablehlo.reshape %53 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %56 = stablehlo.transpose %55, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %57 = stablehlo.reshape %52 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %58 = stablehlo.transpose %57, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %59 = stablehlo.reshape %54 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %60 = stablehlo.transpose %59, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %61 = stablehlo.transpose %56, dims = [0, 1, 3, 2] : (tensor<64x4x32x16xf32>) -> tensor<64x4x16x32xf32>
    %62 = stablehlo.reshape %58 : (tensor<64x4x32x16xf32>) -> tensor<256x32x16xf32>
    %63 = stablehlo.reshape %61 : (tensor<64x4x16x32xf32>) -> tensor<256x16x32xf32>
    %64 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2] : (tensor<256x16x32xf32>) -> tensor<256x16x32xf32>
    %65 = stablehlo.dot_general %62, %64, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x32x16xf32>, tensor<256x16x32xf32>) -> tensor<256x32x32xf32>
    %66 = stablehlo.reshape %65 : (tensor<256x32x32xf32>) -> tensor<64x4x32x32xf32>
    %67 = stablehlo.convert %cst_15 : (tensor<1xf64>) -> tensor<1xf32>
    %68 = stablehlo.reshape %67 : (tensor<1xf32>) -> tensor<f32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f32>) -> tensor<64x4x32x32xf32>
    %70 = stablehlo.multiply %66, %69 : tensor<64x4x32x32xf32>
    %71 = stablehlo.iota dim = 1 : tensor<32x32xi64>
    %72 = stablehlo.iota dim = 0 : tensor<32x32xi64>
    %73 = stablehlo.add %72, %c_1 : tensor<32x32xi64>
    %74 = stablehlo.compare  LE, %71, %73,  SIGNED : (tensor<32x32xi64>, tensor<32x32xi64>) -> tensor<32x32xi1>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1] : (tensor<32x32xi1>) -> tensor<32x32xi1>
    %76 = stablehlo.select %75, %cst_8, %cst_10 : tensor<32x32xi1>, tensor<32x32xf32>
    %77 = stablehlo.reshape %76 : (tensor<32x32xf32>) -> tensor<1x1x32x32xf32>
    %78 = stablehlo.convert %c_6 : (tensor<i64>) -> tensor<f32>
    %79 = stablehlo.broadcast_in_dim %78, dims = [] : (tensor<f32>) -> tensor<1x1x32x32xf32>
    %80 = stablehlo.compare  EQ, %77, %79,  FLOAT : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xi1>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2, 3] : (tensor<1x1x32x32xi1>) -> tensor<64x4x32x32xi1>
    %82 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x4x32x32xf32>
    %83 = stablehlo.broadcast_in_dim %70, dims = [0, 1, 2, 3] : (tensor<64x4x32x32xf32>) -> tensor<64x4x32x32xf32>
    %84 = stablehlo.select %81, %82, %83 : tensor<64x4x32x32xi1>, tensor<64x4x32x32xf32>
    %85 = stablehlo.reduce(%84 init: %cst_7) applies stablehlo.maximum across dimensions = [3] : (tensor<64x4x32x32xf32>, tensor<f32>) -> tensor<64x4x32xf32>
    %86 = stablehlo.reshape %85 : (tensor<64x4x32xf32>) -> tensor<64x4x32x1xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1, 2, 3] : (tensor<64x4x32x1xf32>) -> tensor<64x4x32x32xf32>
    %88 = stablehlo.subtract %84, %87 : tensor<64x4x32x32xf32>
    %89 = stablehlo.exponential %88 : tensor<64x4x32x32xf32>
    %90 = stablehlo.reduce(%89 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x4x32x32xf32>, tensor<f32>) -> tensor<64x4x32xf32>
    %91 = stablehlo.reshape %90 : (tensor<64x4x32xf32>) -> tensor<64x4x32x1xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 3] : (tensor<64x4x32x1xf32>) -> tensor<64x4x32x32xf32>
    %93 = stablehlo.divide %89, %92 : tensor<64x4x32x32xf32>
    %94 = stablehlo.reshape %93 : (tensor<64x4x32x32xf32>) -> tensor<256x32x32xf32>
    %95 = stablehlo.reshape %60 : (tensor<64x4x32x16xf32>) -> tensor<256x32x16xf32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2] : (tensor<256x32x16xf32>) -> tensor<256x32x16xf32>
    %97 = stablehlo.dot_general %94, %96, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x32x32xf32>, tensor<256x32x16xf32>) -> tensor<256x32x16xf32>
    %98 = stablehlo.reshape %97 : (tensor<256x32x16xf32>) -> tensor<64x4x32x16xf32>
    %99 = stablehlo.transpose %98, dims = [0, 2, 1, 3] : (tensor<64x4x32x16xf32>) -> tensor<64x32x4x16xf32>
    %100 = stablehlo.reshape %99 : (tensor<64x32x4x16xf32>) -> tensor<64x32x64xf32>
    %101 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %102 = stablehlo.reshape %100 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %103 = stablehlo.dot_general %102, %101, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %104 = stablehlo.reshape %103 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %105 = stablehlo.add %14, %104 : tensor<64x32x64xf32>
    %106 = stablehlo.convert %105 : (tensor<64x32x64xf32>) -> tensor<64x32x64xf64>
    %107 = stablehlo.reduce(%106 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf64>, tensor<f64>) -> tensor<64x32xf64>
    %108 = stablehlo.reshape %107 : (tensor<64x32xf64>) -> tensor<64x32x1xf64>
    %109 = stablehlo.divide %108, %20 : tensor<64x32x1xf64>
    %110 = stablehlo.broadcast_in_dim %109, dims = [0, 1, 2] : (tensor<64x32x1xf64>) -> tensor<64x32x64xf64>
    %111 = stablehlo.subtract %106, %110 : tensor<64x32x64xf64>
    %112 = stablehlo.multiply %111, %111 : tensor<64x32x64xf64>
    %113 = stablehlo.reduce(%112 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf64>, tensor<f64>) -> tensor<64x32xf64>
    %114 = stablehlo.reshape %113 : (tensor<64x32xf64>) -> tensor<64x32x1xf64>
    %115 = stablehlo.divide %114, %20 : tensor<64x32x1xf64>
    %116 = stablehlo.convert %115 : (tensor<64x32x1xf64>) -> tensor<64x32x1xf32>
    %117 = stablehlo.reduce(%105 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %118 = stablehlo.reshape %117 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %119 = stablehlo.divide %118, %33 : tensor<64x32x1xf32>
    %120 = stablehlo.add %116, %37 : tensor<64x32x1xf32>
    %121 = stablehlo.rsqrt %120 : tensor<64x32x1xf32>
    %122 = stablehlo.broadcast_in_dim %119, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %123 = stablehlo.subtract %105, %122 : tensor<64x32x64xf32>
    %124 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %125 = stablehlo.multiply %123, %124 : tensor<64x32x64xf32>
    %126 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<64xf32>) -> tensor<64x32x64xf32>
    %127 = stablehlo.multiply %125, %126 : tensor<64x32x64xf32>
    %128 = stablehlo.broadcast_in_dim %arg7, dims = [2] : (tensor<64xf32>) -> tensor<64x32x64xf32>
    %129 = stablehlo.add %127, %128 : tensor<64x32x64xf32>
    %130 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
    %131 = stablehlo.reshape %129 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %132 = stablehlo.dot_general %131, %130, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x256xf32>) -> tensor<2048x256xf32>
    %133 = stablehlo.reshape %132 : (tensor<2048x256xf32>) -> tensor<64x32x256xf32>
    %134 = stablehlo.maximum %133, %cst_11 : tensor<64x32x256xf32>
    %135 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<64x256xf32>) -> tensor<256x64xf32>
    %136 = stablehlo.reshape %134 : (tensor<64x32x256xf32>) -> tensor<2048x256xf32>
    %137 = stablehlo.dot_general %136, %135, contracting_dims = [1] x [0] : (tensor<2048x256xf32>, tensor<256x64xf32>) -> tensor<2048x64xf32>
    %138 = stablehlo.reshape %137 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %139 = stablehlo.add %105, %138 : tensor<64x32x64xf32>
    %140 = stablehlo.convert %139 : (tensor<64x32x64xf32>) -> tensor<64x32x64xf64>
    %141 = stablehlo.reduce(%140 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf64>, tensor<f64>) -> tensor<64x32xf64>
    %142 = stablehlo.reshape %141 : (tensor<64x32xf64>) -> tensor<64x32x1xf64>
    %143 = stablehlo.divide %142, %20 : tensor<64x32x1xf64>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<64x32x1xf64>) -> tensor<64x32x64xf64>
    %145 = stablehlo.subtract %140, %144 : tensor<64x32x64xf64>
    %146 = stablehlo.multiply %145, %145 : tensor<64x32x64xf64>
    %147 = stablehlo.reduce(%146 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf64>, tensor<f64>) -> tensor<64x32xf64>
    %148 = stablehlo.reshape %147 : (tensor<64x32xf64>) -> tensor<64x32x1xf64>
    %149 = stablehlo.divide %148, %20 : tensor<64x32x1xf64>
    %150 = stablehlo.convert %149 : (tensor<64x32x1xf64>) -> tensor<64x32x1xf32>
    %151 = stablehlo.reduce(%139 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %152 = stablehlo.reshape %151 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %153 = stablehlo.divide %152, %33 : tensor<64x32x1xf32>
    %154 = stablehlo.add %150, %37 : tensor<64x32x1xf32>
    %155 = stablehlo.rsqrt %154 : tensor<64x32x1xf32>
    %156 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %157 = stablehlo.subtract %139, %156 : tensor<64x32x64xf32>
    %158 = stablehlo.broadcast_in_dim %155, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %159 = stablehlo.multiply %157, %158 : tensor<64x32x64xf32>
    %160 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<64xf32>) -> tensor<64x32x64xf32>
    %161 = stablehlo.multiply %159, %160 : tensor<64x32x64xf32>
    %162 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<64xf32>) -> tensor<64x32x64xf32>
    %163 = stablehlo.add %161, %162 : tensor<64x32x64xf32>
    %164 = stablehlo.transpose %arg12, dims = [1, 0] : (tensor<192x64xf32>) -> tensor<64x192xf32>
    %165 = stablehlo.reshape %163 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %166 = stablehlo.dot_general %165, %164, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x192xf32>) -> tensor<2048x192xf32>
    %167 = stablehlo.reshape %166 : (tensor<2048x192xf32>) -> tensor<64x32x192xf32>
    %168 = stablehlo.slice %167 [0:64, 0:32, 0:64] : (tensor<64x32x192xf32>) -> tensor<64x32x64xf32>
    %169 = stablehlo.slice %167 [0:64, 0:32, 64:128] : (tensor<64x32x192xf32>) -> tensor<64x32x64xf32>
    %170 = stablehlo.slice %167 [0:64, 0:32, 128:192] : (tensor<64x32x192xf32>) -> tensor<64x32x64xf32>
    %171 = stablehlo.reshape %169 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %172 = stablehlo.transpose %171, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %173 = stablehlo.reshape %168 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %174 = stablehlo.transpose %173, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %175 = stablehlo.reshape %170 : (tensor<64x32x64xf32>) -> tensor<64x32x4x16xf32>
    %176 = stablehlo.transpose %175, dims = [0, 2, 1, 3] : (tensor<64x32x4x16xf32>) -> tensor<64x4x32x16xf32>
    %177 = stablehlo.transpose %172, dims = [0, 1, 3, 2] : (tensor<64x4x32x16xf32>) -> tensor<64x4x16x32xf32>
    %178 = stablehlo.reshape %174 : (tensor<64x4x32x16xf32>) -> tensor<256x32x16xf32>
    %179 = stablehlo.reshape %177 : (tensor<64x4x16x32xf32>) -> tensor<256x16x32xf32>
    %180 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2] : (tensor<256x16x32xf32>) -> tensor<256x16x32xf32>
    %181 = stablehlo.dot_general %178, %180, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x32x16xf32>, tensor<256x16x32xf32>) -> tensor<256x32x32xf32>
    %182 = stablehlo.reshape %181 : (tensor<256x32x32xf32>) -> tensor<64x4x32x32xf32>
    %183 = stablehlo.multiply %182, %69 : tensor<64x4x32x32xf32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0, 1, 2, 3] : (tensor<64x4x32x32xf32>) -> tensor<64x4x32x32xf32>
    %185 = stablehlo.select %81, %82, %184 : tensor<64x4x32x32xi1>, tensor<64x4x32x32xf32>
    %186 = stablehlo.reduce(%185 init: %cst_7) applies stablehlo.maximum across dimensions = [3] : (tensor<64x4x32x32xf32>, tensor<f32>) -> tensor<64x4x32xf32>
    %187 = stablehlo.reshape %186 : (tensor<64x4x32xf32>) -> tensor<64x4x32x1xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<64x4x32x1xf32>) -> tensor<64x4x32x32xf32>
    %189 = stablehlo.subtract %185, %188 : tensor<64x4x32x32xf32>
    %190 = stablehlo.exponential %189 : tensor<64x4x32x32xf32>
    %191 = stablehlo.reduce(%190 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x4x32x32xf32>, tensor<f32>) -> tensor<64x4x32xf32>
    %192 = stablehlo.reshape %191 : (tensor<64x4x32xf32>) -> tensor<64x4x32x1xf32>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<64x4x32x1xf32>) -> tensor<64x4x32x32xf32>
    %194 = stablehlo.divide %190, %193 : tensor<64x4x32x32xf32>
    %195 = stablehlo.reshape %194 : (tensor<64x4x32x32xf32>) -> tensor<256x32x32xf32>
    %196 = stablehlo.reshape %176 : (tensor<64x4x32x16xf32>) -> tensor<256x32x16xf32>
    %197 = stablehlo.broadcast_in_dim %196, dims = [0, 1, 2] : (tensor<256x32x16xf32>) -> tensor<256x32x16xf32>
    %198 = stablehlo.dot_general %195, %197, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<256x32x32xf32>, tensor<256x32x16xf32>) -> tensor<256x32x16xf32>
    %199 = stablehlo.reshape %198 : (tensor<256x32x16xf32>) -> tensor<64x4x32x16xf32>
    %200 = stablehlo.transpose %199, dims = [0, 2, 1, 3] : (tensor<64x4x32x16xf32>) -> tensor<64x32x4x16xf32>
    %201 = stablehlo.reshape %200 : (tensor<64x32x4x16xf32>) -> tensor<64x32x64xf32>
    %202 = stablehlo.transpose %arg13, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %203 = stablehlo.reshape %201 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %204 = stablehlo.dot_general %203, %202, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %205 = stablehlo.reshape %204 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %206 = stablehlo.add %139, %205 : tensor<64x32x64xf32>
    %207 = stablehlo.convert %206 : (tensor<64x32x64xf32>) -> tensor<64x32x64xf64>
    %208 = stablehlo.reduce(%207 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf64>, tensor<f64>) -> tensor<64x32xf64>
    %209 = stablehlo.reshape %208 : (tensor<64x32xf64>) -> tensor<64x32x1xf64>
    %210 = stablehlo.divide %209, %20 : tensor<64x32x1xf64>
    %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<64x32x1xf64>) -> tensor<64x32x64xf64>
    %212 = stablehlo.subtract %207, %211 : tensor<64x32x64xf64>
    %213 = stablehlo.multiply %212, %212 : tensor<64x32x64xf64>
    %214 = stablehlo.reduce(%213 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf64>, tensor<f64>) -> tensor<64x32xf64>
    %215 = stablehlo.reshape %214 : (tensor<64x32xf64>) -> tensor<64x32x1xf64>
    %216 = stablehlo.divide %215, %20 : tensor<64x32x1xf64>
    %217 = stablehlo.convert %216 : (tensor<64x32x1xf64>) -> tensor<64x32x1xf32>
    %218 = stablehlo.reduce(%206 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %219 = stablehlo.reshape %218 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %220 = stablehlo.divide %219, %33 : tensor<64x32x1xf32>
    %221 = stablehlo.add %217, %37 : tensor<64x32x1xf32>
    %222 = stablehlo.rsqrt %221 : tensor<64x32x1xf32>
    %223 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %224 = stablehlo.subtract %206, %223 : tensor<64x32x64xf32>
    %225 = stablehlo.broadcast_in_dim %222, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %226 = stablehlo.multiply %224, %225 : tensor<64x32x64xf32>
    %227 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<64xf32>) -> tensor<64x32x64xf32>
    %228 = stablehlo.multiply %226, %227 : tensor<64x32x64xf32>
    %229 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<64xf32>) -> tensor<64x32x64xf32>
    %230 = stablehlo.add %228, %229 : tensor<64x32x64xf32>
    %231 = stablehlo.transpose %arg16, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
    %232 = stablehlo.reshape %230 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %233 = stablehlo.dot_general %232, %231, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x256xf32>) -> tensor<2048x256xf32>
    %234 = stablehlo.reshape %233 : (tensor<2048x256xf32>) -> tensor<64x32x256xf32>
    %235 = stablehlo.maximum %234, %cst_11 : tensor<64x32x256xf32>
    %236 = stablehlo.transpose %arg17, dims = [1, 0] : (tensor<64x256xf32>) -> tensor<256x64xf32>
    %237 = stablehlo.reshape %235 : (tensor<64x32x256xf32>) -> tensor<2048x256xf32>
    %238 = stablehlo.dot_general %237, %236, contracting_dims = [1] x [0] : (tensor<2048x256xf32>, tensor<256x64xf32>) -> tensor<2048x64xf32>
    %239 = stablehlo.reshape %238 : (tensor<2048x64xf32>) -> tensor<64x32x64xf32>
    %240 = stablehlo.add %206, %239 : tensor<64x32x64xf32>
    %241 = stablehlo.convert %240 : (tensor<64x32x64xf32>) -> tensor<64x32x64xf64>
    %242 = stablehlo.reduce(%241 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf64>, tensor<f64>) -> tensor<64x32xf64>
    %243 = stablehlo.reshape %242 : (tensor<64x32xf64>) -> tensor<64x32x1xf64>
    %244 = stablehlo.divide %243, %20 : tensor<64x32x1xf64>
    %245 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2] : (tensor<64x32x1xf64>) -> tensor<64x32x64xf64>
    %246 = stablehlo.subtract %241, %245 : tensor<64x32x64xf64>
    %247 = stablehlo.multiply %246, %246 : tensor<64x32x64xf64>
    %248 = stablehlo.reduce(%247 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf64>, tensor<f64>) -> tensor<64x32xf64>
    %249 = stablehlo.reshape %248 : (tensor<64x32xf64>) -> tensor<64x32x1xf64>
    %250 = stablehlo.divide %249, %20 : tensor<64x32x1xf64>
    %251 = stablehlo.convert %250 : (tensor<64x32x1xf64>) -> tensor<64x32x1xf32>
    %252 = stablehlo.reduce(%240 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x32x64xf32>, tensor<f32>) -> tensor<64x32xf32>
    %253 = stablehlo.reshape %252 : (tensor<64x32xf32>) -> tensor<64x32x1xf32>
    %254 = stablehlo.divide %253, %33 : tensor<64x32x1xf32>
    %255 = stablehlo.add %251, %37 : tensor<64x32x1xf32>
    %256 = stablehlo.rsqrt %255 : tensor<64x32x1xf32>
    %257 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %258 = stablehlo.subtract %240, %257 : tensor<64x32x64xf32>
    %259 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<64x32x1xf32>) -> tensor<64x32x64xf32>
    %260 = stablehlo.multiply %258, %259 : tensor<64x32x64xf32>
    %261 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<64xf32>) -> tensor<64x32x64xf32>
    %262 = stablehlo.multiply %260, %261 : tensor<64x32x64xf32>
    %263 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<64xf32>) -> tensor<64x32x64xf32>
    %264 = stablehlo.add %262, %263 : tensor<64x32x64xf32>
    %265 = stablehlo.reshape %264 : (tensor<64x32x64xf32>) -> tensor<2048x64xf32>
    %266 = stablehlo.transpose %arg20, dims = [1, 0] : (tensor<65x64xf32>) -> tensor<64x65xf32>
    %267 = stablehlo.dot_general %265, %266, contracting_dims = [1] x [0] : (tensor<2048x64xf32>, tensor<64x65xf32>) -> tensor<2048x65xf32>
    %268 = stablehlo.convert %cst_12 : (tensor<1xi64>) -> tensor<1xf32>
    %269 = stablehlo.reshape %268 : (tensor<1xf32>) -> tensor<f32>
    %270 = stablehlo.broadcast_in_dim %269, dims = [] : (tensor<f32>) -> tensor<2048x65xf32>
    %271 = stablehlo.multiply %267, %270 : tensor<2048x65xf32>
    %272 = stablehlo.broadcast_in_dim %269, dims = [] : (tensor<f32>) -> tensor<65xf32>
    %273 = stablehlo.multiply %arg21, %272 : tensor<65xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [1] : (tensor<65xf32>) -> tensor<2048x65xf32>
    %275 = stablehlo.add %271, %274 : tensor<2048x65xf32>
    %276 = stablehlo.reshape %275 : (tensor<2048x65xf32>) -> tensor<64x32x65xf32>
    %277 = stablehlo.reshape %276 : (tensor<64x32x65xf32>) -> tensor<2048x65xf32>
    %278 = stablehlo.reshape %arg23 : (tensor<64x32xi64>) -> tensor<2048xi64>
    %279 = stablehlo.reduce(%277 init: %cst_7) applies stablehlo.maximum across dimensions = [1] : (tensor<2048x65xf32>, tensor<f32>) -> tensor<2048xf32>
    %280 = stablehlo.reshape %279 : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %281 = stablehlo.broadcast_in_dim %280, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x65xf32>
    %282 = stablehlo.subtract %277, %281 : tensor<2048x65xf32>
    %283 = stablehlo.exponential %282 : tensor<2048x65xf32>
    %284 = stablehlo.reduce(%283 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2048x65xf32>, tensor<f32>) -> tensor<2048xf32>
    %285 = stablehlo.reshape %284 : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %286 = stablehlo.log %285 : tensor<2048x1xf32>
    %287 = stablehlo.broadcast_in_dim %286, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x65xf32>
    %288 = stablehlo.subtract %282, %287 : tensor<2048x65xf32>
    %289 = stablehlo.compare  NE, %278, %c_0,  SIGNED : (tensor<2048xi64>, tensor<2048xi64>) -> tensor<2048xi1>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0] : (tensor<2048xi1>) -> tensor<2048xi1>
    %291 = stablehlo.broadcast_in_dim %278, dims = [0] : (tensor<2048xi64>) -> tensor<2048xi64>
    %292 = stablehlo.select %290, %291, %c : tensor<2048xi1>, tensor<2048xi64>
    %293 = stablehlo.reshape %292 : (tensor<2048xi64>) -> tensor<2048x1xi64>
    %294 = stablehlo.iota dim = 0 : tensor<2048x1x1xi64>
    %295 = stablehlo.reshape %293 : (tensor<2048x1xi64>) -> tensor<2048x1x1xi64>
    %296 = stablehlo.concatenate %294, %295, dim = 2 : (tensor<2048x1x1xi64>, tensor<2048x1x1xi64>) -> tensor<2048x1x2xi64>
    %297 = "stablehlo.gather"(%288, %296) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<2048x65xf32>, tensor<2048x1x2xi64>) -> tensor<2048x1xf32>
    %298 = stablehlo.reshape %297 : (tensor<2048x1xf32>) -> tensor<2048xf32>
    %299 = stablehlo.negate %298 : tensor<2048xf32>
    %300 = stablehlo.broadcast_in_dim %299, dims = [0] : (tensor<2048xf32>) -> tensor<2048xf32>
    %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %302 = stablehlo.select %290, %300, %301 : tensor<2048xi1>, tensor<2048xf32>
    %303 = stablehlo.convert %289 : (tensor<2048xi1>) -> tensor<2048xi64>
    %304 = stablehlo.reduce(%303 init: %c_6) applies stablehlo.add across dimensions = [0] : (tensor<2048xi64>, tensor<i64>) -> tensor<i64>
    %305 = stablehlo.convert %304 : (tensor<i64>) -> tensor<f32>
    %306 = stablehlo.reduce(%302 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2048xf32>, tensor<f32>) -> tensor<f32>
    %307 = stablehlo.divide %306, %305 : tensor<f32>
    return %307 : tensor<f32>
  }
}
