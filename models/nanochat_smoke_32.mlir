module {
  func.func @main(%arg0: tensor<65536x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128x128xf32>, %arg4: tensor<128x128xf32>, %arg5: tensor<512x128xf32>, %arg6: tensor<128x512xf32>, %arg7: tensor<128x128xf32>, %arg8: tensor<128x128xf32>, %arg9: tensor<128x128xf32>, %arg10: tensor<128x128xf32>, %arg11: tensor<512x128xf32>, %arg12: tensor<128x512xf32>, %arg13: tensor<65536x128xf32>, %arg14: tensor<1x32xi64>, %arg15: tensor<1x32xi64>) -> tensor<f32> {
    %c = stablehlo.constant dense<-1> : tensor<32xi64>
    %c_0 = stablehlo.constant dense<1> : tensor<32x32xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<32xi64>
    %c_2 = stablehlo.constant dense<1> : tensor<32xi64>
    %c_3 = stablehlo.constant dense<32> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_5 = stablehlo.constant dense<0> : tensor<i64>
    %cst_6 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_7 = stablehlo.constant dense<true> : tensor<32x32xi1>
    %cst_8 = stablehlo.constant dense_resource<torch_tensor_1_32_1_16_torch.bfloat16_1> : tensor<1x32x1x16xbf16>
    %cst_9 = stablehlo.constant dense_resource<torch_tensor_1_32_1_16_torch.bfloat16> : tensor<1x32x1x16xbf16>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<1x32x512xf32>
    %cst_11 = arith.constant dense<2> : tensor<1xi64>
    %cst_12 = arith.constant dense<128> : tensor<1xi64>
    %cst_13 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_14 = arith.constant dense<32> : tensor<1xi64>
    %cst_15 = arith.constant dense<0.17677669529663687> : tensor<1xf64>
    %cst_16 = arith.constant dense<15> : tensor<1xi64>
    %0 = "stablehlo.gather"(%arg0, %arg14) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<65536x128xf32>, tensor<1x32xi64>) -> tensor<1x32x128xf32>
    %1 = stablehlo.convert %0 : tensor<1x32x128xf32>
    %2 = stablehlo.convert %cst_11 : (tensor<1xi64>) -> tensor<1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x32x128xf32>
    %5 = stablehlo.power %1, %4 : tensor<1x32x128xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x32x128xf32>, tensor<f32>) -> tensor<1x32xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %8 = stablehlo.convert %cst_12 : (tensor<1xi64>) -> tensor<1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<1x32x1xf32>
    %11 = stablehlo.divide %7, %10 : tensor<1x32x1xf32>
    %12 = stablehlo.convert %cst_13 : (tensor<1xf64>) -> tensor<1xf32>
    %13 = stablehlo.reshape %12 : (tensor<1xf32>) -> tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<1x32x1xf32>
    %15 = stablehlo.add %11, %14 : tensor<1x32x1xf32>
    %16 = stablehlo.rsqrt %15 : tensor<1x32x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<1x32x1xf32>) -> tensor<1x32x128xf32>
    %18 = stablehlo.multiply %1, %17 : tensor<1x32x128xf32>
    %19 = stablehlo.power %18, %4 : tensor<1x32x128xf32>
    %20 = stablehlo.reduce(%19 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x32x128xf32>, tensor<f32>) -> tensor<1x32xf32>
    %21 = stablehlo.reshape %20 : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %22 = stablehlo.divide %21, %10 : tensor<1x32x1xf32>
    %23 = stablehlo.add %22, %14 : tensor<1x32x1xf32>
    %24 = stablehlo.rsqrt %23 : tensor<1x32x1xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2] : (tensor<1x32x1xf32>) -> tensor<1x32x128xf32>
    %26 = stablehlo.multiply %18, %25 : tensor<1x32x128xf32>
    %27 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %28 = stablehlo.reshape %26 : (tensor<1x32x128xf32>) -> tensor<32x128xf32>
    %29 = stablehlo.dot_general %28, %27, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x128xf32>) -> tensor<32x128xf32>
    %30 = stablehlo.reshape %29 : (tensor<32x128xf32>) -> tensor<1x32x128xf32>
    %31 = stablehlo.reshape %30 : (tensor<1x32x128xf32>) -> tensor<1x32x4x32xf32>
    %32 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %33 = stablehlo.dot_general %28, %32, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x128xf32>) -> tensor<32x128xf32>
    %34 = stablehlo.reshape %33 : (tensor<32x128xf32>) -> tensor<1x32x128xf32>
    %35 = stablehlo.reshape %34 : (tensor<1x32x128xf32>) -> tensor<1x32x4x32xf32>
    %36 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %37 = stablehlo.dot_general %28, %36, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x128xf32>) -> tensor<32x128xf32>
    %38 = stablehlo.reshape %37 : (tensor<32x128xf32>) -> tensor<1x32x128xf32>
    %39 = stablehlo.reshape %38 : (tensor<1x32x128xf32>) -> tensor<1x32x4x32xf32>
    %40 = stablehlo.slice %31 [0:1, 0:32, 0:4, 0:16] : (tensor<1x32x4x32xf32>) -> tensor<1x32x4x16xf32>
    %41 = stablehlo.slice %31 [0:1, 0:32, 0:4, 16:32] : (tensor<1x32x4x32xf32>) -> tensor<1x32x4x16xf32>
    %42 = stablehlo.convert %cst_9 : (tensor<1x32x1x16xbf16>) -> tensor<1x32x1x16xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2, 3] : (tensor<1x32x1x16xf32>) -> tensor<1x32x4x16xf32>
    %44 = stablehlo.multiply %40, %43 : tensor<1x32x4x16xf32>
    %45 = stablehlo.convert %cst_8 : (tensor<1x32x1x16xbf16>) -> tensor<1x32x1x16xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x32x1x16xf32>) -> tensor<1x32x4x16xf32>
    %47 = stablehlo.multiply %41, %46 : tensor<1x32x4x16xf32>
    %48 = stablehlo.add %44, %47 : tensor<1x32x4x16xf32>
    %49 = stablehlo.negate %cst_8 : tensor<1x32x1x16xbf16>
    %50 = stablehlo.convert %49 : (tensor<1x32x1x16xbf16>) -> tensor<1x32x1x16xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2, 3] : (tensor<1x32x1x16xf32>) -> tensor<1x32x4x16xf32>
    %52 = stablehlo.multiply %40, %51 : tensor<1x32x4x16xf32>
    %53 = stablehlo.multiply %41, %43 : tensor<1x32x4x16xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x32x4x16xf32>
    %55 = stablehlo.concatenate %48, %54, dim = 3 : (tensor<1x32x4x16xf32>, tensor<1x32x4x16xf32>) -> tensor<1x32x4x32xf32>
    %56 = stablehlo.slice %35 [0:1, 0:32, 0:4, 0:16] : (tensor<1x32x4x32xf32>) -> tensor<1x32x4x16xf32>
    %57 = stablehlo.slice %35 [0:1, 0:32, 0:4, 16:32] : (tensor<1x32x4x32xf32>) -> tensor<1x32x4x16xf32>
    %58 = stablehlo.multiply %56, %43 : tensor<1x32x4x16xf32>
    %59 = stablehlo.multiply %57, %46 : tensor<1x32x4x16xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x32x4x16xf32>
    %61 = stablehlo.multiply %56, %51 : tensor<1x32x4x16xf32>
    %62 = stablehlo.multiply %57, %43 : tensor<1x32x4x16xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x32x4x16xf32>
    %64 = stablehlo.concatenate %60, %63, dim = 3 : (tensor<1x32x4x16xf32>, tensor<1x32x4x16xf32>) -> tensor<1x32x4x32xf32>
    %65 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x32x4x32xf32>
    %66 = stablehlo.power %55, %65 : tensor<1x32x4x32xf32>
    %67 = stablehlo.reduce(%66 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x32x4x32xf32>, tensor<f32>) -> tensor<1x32x4xf32>
    %68 = stablehlo.reshape %67 : (tensor<1x32x4xf32>) -> tensor<1x32x4x1xf32>
    %69 = stablehlo.convert %cst_14 : (tensor<1xi64>) -> tensor<1xf32>
    %70 = stablehlo.reshape %69 : (tensor<1xf32>) -> tensor<f32>
    %71 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<f32>) -> tensor<1x32x4x1xf32>
    %72 = stablehlo.divide %68, %71 : tensor<1x32x4x1xf32>
    %73 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<1x32x4x1xf32>
    %74 = stablehlo.add %72, %73 : tensor<1x32x4x1xf32>
    %75 = stablehlo.rsqrt %74 : tensor<1x32x4x1xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x32x4x1xf32>) -> tensor<1x32x4x32xf32>
    %77 = stablehlo.multiply %55, %76 : tensor<1x32x4x32xf32>
    %78 = stablehlo.power %64, %65 : tensor<1x32x4x32xf32>
    %79 = stablehlo.reduce(%78 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x32x4x32xf32>, tensor<f32>) -> tensor<1x32x4xf32>
    %80 = stablehlo.reshape %79 : (tensor<1x32x4xf32>) -> tensor<1x32x4x1xf32>
    %81 = stablehlo.divide %80, %71 : tensor<1x32x4x1xf32>
    %82 = stablehlo.add %81, %73 : tensor<1x32x4x1xf32>
    %83 = stablehlo.rsqrt %82 : tensor<1x32x4x1xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x32x4x1xf32>) -> tensor<1x32x4x32xf32>
    %85 = stablehlo.multiply %64, %84 : tensor<1x32x4x32xf32>
    %86 = stablehlo.transpose %77, dims = [0, 2, 1, 3] : (tensor<1x32x4x32xf32>) -> tensor<1x4x32x32xf32>
    %87 = stablehlo.transpose %85, dims = [0, 2, 1, 3] : (tensor<1x32x4x32xf32>) -> tensor<1x4x32x32xf32>
    %88 = stablehlo.transpose %39, dims = [0, 2, 1, 3] : (tensor<1x32x4x32xf32>) -> tensor<1x4x32x32xf32>
    %89 = stablehlo.transpose %87, dims = [0, 1, 3, 2] : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %90 = stablehlo.reshape %86 : (tensor<1x4x32x32xf32>) -> tensor<4x32x32xf32>
    %91 = stablehlo.reshape %89 : (tensor<1x4x32x32xf32>) -> tensor<4x32x32xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2] : (tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %93 = stablehlo.dot_general %90, %92, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x32x32xf32>, tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %94 = stablehlo.reshape %93 : (tensor<4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %95 = stablehlo.convert %cst_15 : (tensor<1xf64>) -> tensor<1xf32>
    %96 = stablehlo.reshape %95 : (tensor<1xf32>) -> tensor<f32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<f32>) -> tensor<1x4x32x32xf32>
    %98 = stablehlo.multiply %94, %97 : tensor<1x4x32x32xf32>
    %99 = stablehlo.convert %c_4 : (tensor<i64>) -> tensor<f64>
    %100 = stablehlo.convert %c_3 : (tensor<i64>) -> tensor<f64>
    %101 = stablehlo.divide %100, %99 : tensor<f64>
    %102 = stablehlo.ceil %101 : tensor<f64>
    %103 = stablehlo.convert %102 : (tensor<f64>) -> tensor<i64>
    %104 = stablehlo.reshape %103 : (tensor<i64>) -> tensor<1xi64>
    %105 = stablehlo.dynamic_iota %104, dim = 0 : (tensor<1xi64>) -> tensor<32xi64>
    %106 = stablehlo.multiply %105, %c_2 : tensor<32xi64>
    %107 = stablehlo.add %106, %c_1 : tensor<32xi64>
    %108 = stablehlo.reshape %107 : (tensor<32xi64>) -> tensor<1x32xi64>
    %109 = stablehlo.reshape %107 : (tensor<32xi64>) -> tensor<32x1xi64>
    %110 = stablehlo.broadcast_in_dim %108, dims = [0, 1] : (tensor<1x32xi64>) -> tensor<32x32xi64>
    %111 = stablehlo.broadcast_in_dim %109, dims = [0, 1] : (tensor<32x1xi64>) -> tensor<32x32xi64>
    %112 = stablehlo.subtract %110, %111 : tensor<32x32xi64>
    %113 = stablehlo.compare  GE, %112, %c_0,  SIGNED : (tensor<32x32xi64>, tensor<32x32xi64>) -> tensor<32x32xi1>
    %114 = stablehlo.and %113, %c_7 : tensor<32x32xi1>
    %115 = stablehlo.reshape %114 : (tensor<32x32xi1>) -> tensor<1x1x32x32xi1>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0, 1, 2, 3] : (tensor<1x1x32x32xi1>) -> tensor<1x4x32x32xi1>
    %117 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x4x32x32xf32>
    %118 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2, 3] : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %119 = stablehlo.select %116, %117, %118 : tensor<1x4x32x32xi1>, tensor<1x4x32x32xf32>
    %120 = stablehlo.reduce(%119 init: %cst_6) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x32x32xf32>, tensor<f32>) -> tensor<1x4x32xf32>
    %121 = stablehlo.reshape %120 : (tensor<1x4x32xf32>) -> tensor<1x4x32x1xf32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2, 3] : (tensor<1x4x32x1xf32>) -> tensor<1x4x32x32xf32>
    %123 = stablehlo.subtract %119, %122 : tensor<1x4x32x32xf32>
    %124 = stablehlo.exponential %123 : tensor<1x4x32x32xf32>
    %125 = stablehlo.reduce(%124 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x4x32x32xf32>, tensor<f32>) -> tensor<1x4x32xf32>
    %126 = stablehlo.reshape %125 : (tensor<1x4x32xf32>) -> tensor<1x4x32x1xf32>
    %127 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2, 3] : (tensor<1x4x32x1xf32>) -> tensor<1x4x32x32xf32>
    %128 = stablehlo.divide %124, %127 : tensor<1x4x32x32xf32>
    %129 = stablehlo.reshape %128 : (tensor<1x4x32x32xf32>) -> tensor<4x32x32xf32>
    %130 = stablehlo.reshape %88 : (tensor<1x4x32x32xf32>) -> tensor<4x32x32xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %132 = stablehlo.dot_general %129, %131, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x32x32xf32>, tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %133 = stablehlo.reshape %132 : (tensor<4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %134 = stablehlo.transpose %133, dims = [0, 2, 1, 3] : (tensor<1x4x32x32xf32>) -> tensor<1x32x4x32xf32>
    %135 = stablehlo.reshape %134 : (tensor<1x32x4x32xf32>) -> tensor<1x32x128xf32>
    %136 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %137 = stablehlo.reshape %135 : (tensor<1x32x128xf32>) -> tensor<32x128xf32>
    %138 = stablehlo.dot_general %137, %136, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x128xf32>) -> tensor<32x128xf32>
    %139 = stablehlo.reshape %138 : (tensor<32x128xf32>) -> tensor<1x32x128xf32>
    %140 = stablehlo.add %18, %139 : tensor<1x32x128xf32>
    %141 = stablehlo.power %140, %4 : tensor<1x32x128xf32>
    %142 = stablehlo.reduce(%141 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x32x128xf32>, tensor<f32>) -> tensor<1x32xf32>
    %143 = stablehlo.reshape %142 : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %144 = stablehlo.divide %143, %10 : tensor<1x32x1xf32>
    %145 = stablehlo.add %144, %14 : tensor<1x32x1xf32>
    %146 = stablehlo.rsqrt %145 : tensor<1x32x1xf32>
    %147 = stablehlo.broadcast_in_dim %146, dims = [0, 1, 2] : (tensor<1x32x1xf32>) -> tensor<1x32x128xf32>
    %148 = stablehlo.multiply %140, %147 : tensor<1x32x128xf32>
    %149 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %150 = stablehlo.reshape %148 : (tensor<1x32x128xf32>) -> tensor<32x128xf32>
    %151 = stablehlo.dot_general %150, %149, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x512xf32>) -> tensor<32x512xf32>
    %152 = stablehlo.reshape %151 : (tensor<32x512xf32>) -> tensor<1x32x512xf32>
    %153 = stablehlo.maximum %152, %cst_10 : tensor<1x32x512xf32>
    %154 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x32x512xf32>
    %155 = stablehlo.power %153, %154 : tensor<1x32x512xf32>
    %156 = stablehlo.transpose %arg6, dims = [1, 0] : (tensor<128x512xf32>) -> tensor<512x128xf32>
    %157 = stablehlo.reshape %155 : (tensor<1x32x512xf32>) -> tensor<32x512xf32>
    %158 = stablehlo.dot_general %157, %156, contracting_dims = [1] x [0] : (tensor<32x512xf32>, tensor<512x128xf32>) -> tensor<32x128xf32>
    %159 = stablehlo.reshape %158 : (tensor<32x128xf32>) -> tensor<1x32x128xf32>
    %160 = stablehlo.add %140, %159 : tensor<1x32x128xf32>
    %161 = stablehlo.power %160, %4 : tensor<1x32x128xf32>
    %162 = stablehlo.reduce(%161 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x32x128xf32>, tensor<f32>) -> tensor<1x32xf32>
    %163 = stablehlo.reshape %162 : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %164 = stablehlo.divide %163, %10 : tensor<1x32x1xf32>
    %165 = stablehlo.add %164, %14 : tensor<1x32x1xf32>
    %166 = stablehlo.rsqrt %165 : tensor<1x32x1xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2] : (tensor<1x32x1xf32>) -> tensor<1x32x128xf32>
    %168 = stablehlo.multiply %160, %167 : tensor<1x32x128xf32>
    %169 = stablehlo.transpose %arg7, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %170 = stablehlo.reshape %168 : (tensor<1x32x128xf32>) -> tensor<32x128xf32>
    %171 = stablehlo.dot_general %170, %169, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x128xf32>) -> tensor<32x128xf32>
    %172 = stablehlo.reshape %171 : (tensor<32x128xf32>) -> tensor<1x32x128xf32>
    %173 = stablehlo.reshape %172 : (tensor<1x32x128xf32>) -> tensor<1x32x4x32xf32>
    %174 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %175 = stablehlo.dot_general %170, %174, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x128xf32>) -> tensor<32x128xf32>
    %176 = stablehlo.reshape %175 : (tensor<32x128xf32>) -> tensor<1x32x128xf32>
    %177 = stablehlo.reshape %176 : (tensor<1x32x128xf32>) -> tensor<1x32x4x32xf32>
    %178 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %179 = stablehlo.dot_general %170, %178, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x128xf32>) -> tensor<32x128xf32>
    %180 = stablehlo.reshape %179 : (tensor<32x128xf32>) -> tensor<1x32x128xf32>
    %181 = stablehlo.reshape %180 : (tensor<1x32x128xf32>) -> tensor<1x32x4x32xf32>
    %182 = stablehlo.slice %173 [0:1, 0:32, 0:4, 0:16] : (tensor<1x32x4x32xf32>) -> tensor<1x32x4x16xf32>
    %183 = stablehlo.slice %173 [0:1, 0:32, 0:4, 16:32] : (tensor<1x32x4x32xf32>) -> tensor<1x32x4x16xf32>
    %184 = stablehlo.multiply %182, %43 : tensor<1x32x4x16xf32>
    %185 = stablehlo.multiply %183, %46 : tensor<1x32x4x16xf32>
    %186 = stablehlo.add %184, %185 : tensor<1x32x4x16xf32>
    %187 = stablehlo.multiply %182, %51 : tensor<1x32x4x16xf32>
    %188 = stablehlo.multiply %183, %43 : tensor<1x32x4x16xf32>
    %189 = stablehlo.add %187, %188 : tensor<1x32x4x16xf32>
    %190 = stablehlo.concatenate %186, %189, dim = 3 : (tensor<1x32x4x16xf32>, tensor<1x32x4x16xf32>) -> tensor<1x32x4x32xf32>
    %191 = stablehlo.slice %177 [0:1, 0:32, 0:4, 0:16] : (tensor<1x32x4x32xf32>) -> tensor<1x32x4x16xf32>
    %192 = stablehlo.slice %177 [0:1, 0:32, 0:4, 16:32] : (tensor<1x32x4x32xf32>) -> tensor<1x32x4x16xf32>
    %193 = stablehlo.multiply %191, %43 : tensor<1x32x4x16xf32>
    %194 = stablehlo.multiply %192, %46 : tensor<1x32x4x16xf32>
    %195 = stablehlo.add %193, %194 : tensor<1x32x4x16xf32>
    %196 = stablehlo.multiply %191, %51 : tensor<1x32x4x16xf32>
    %197 = stablehlo.multiply %192, %43 : tensor<1x32x4x16xf32>
    %198 = stablehlo.add %196, %197 : tensor<1x32x4x16xf32>
    %199 = stablehlo.concatenate %195, %198, dim = 3 : (tensor<1x32x4x16xf32>, tensor<1x32x4x16xf32>) -> tensor<1x32x4x32xf32>
    %200 = stablehlo.power %190, %65 : tensor<1x32x4x32xf32>
    %201 = stablehlo.reduce(%200 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x32x4x32xf32>, tensor<f32>) -> tensor<1x32x4xf32>
    %202 = stablehlo.reshape %201 : (tensor<1x32x4xf32>) -> tensor<1x32x4x1xf32>
    %203 = stablehlo.divide %202, %71 : tensor<1x32x4x1xf32>
    %204 = stablehlo.add %203, %73 : tensor<1x32x4x1xf32>
    %205 = stablehlo.rsqrt %204 : tensor<1x32x4x1xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<1x32x4x1xf32>) -> tensor<1x32x4x32xf32>
    %207 = stablehlo.multiply %190, %206 : tensor<1x32x4x32xf32>
    %208 = stablehlo.power %199, %65 : tensor<1x32x4x32xf32>
    %209 = stablehlo.reduce(%208 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x32x4x32xf32>, tensor<f32>) -> tensor<1x32x4xf32>
    %210 = stablehlo.reshape %209 : (tensor<1x32x4xf32>) -> tensor<1x32x4x1xf32>
    %211 = stablehlo.divide %210, %71 : tensor<1x32x4x1xf32>
    %212 = stablehlo.add %211, %73 : tensor<1x32x4x1xf32>
    %213 = stablehlo.rsqrt %212 : tensor<1x32x4x1xf32>
    %214 = stablehlo.broadcast_in_dim %213, dims = [0, 1, 2, 3] : (tensor<1x32x4x1xf32>) -> tensor<1x32x4x32xf32>
    %215 = stablehlo.multiply %199, %214 : tensor<1x32x4x32xf32>
    %216 = stablehlo.transpose %207, dims = [0, 2, 1, 3] : (tensor<1x32x4x32xf32>) -> tensor<1x4x32x32xf32>
    %217 = stablehlo.transpose %215, dims = [0, 2, 1, 3] : (tensor<1x32x4x32xf32>) -> tensor<1x4x32x32xf32>
    %218 = stablehlo.transpose %181, dims = [0, 2, 1, 3] : (tensor<1x32x4x32xf32>) -> tensor<1x4x32x32xf32>
    %219 = stablehlo.transpose %217, dims = [0, 1, 3, 2] : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %220 = stablehlo.reshape %216 : (tensor<1x4x32x32xf32>) -> tensor<4x32x32xf32>
    %221 = stablehlo.reshape %219 : (tensor<1x4x32x32xf32>) -> tensor<4x32x32xf32>
    %222 = stablehlo.broadcast_in_dim %221, dims = [0, 1, 2] : (tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %223 = stablehlo.dot_general %220, %222, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x32x32xf32>, tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %224 = stablehlo.reshape %223 : (tensor<4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %225 = stablehlo.multiply %224, %97 : tensor<1x4x32x32xf32>
    %226 = stablehlo.broadcast_in_dim %225, dims = [0, 1, 2, 3] : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %227 = stablehlo.select %116, %117, %226 : tensor<1x4x32x32xi1>, tensor<1x4x32x32xf32>
    %228 = stablehlo.reduce(%227 init: %cst_6) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x32x32xf32>, tensor<f32>) -> tensor<1x4x32xf32>
    %229 = stablehlo.reshape %228 : (tensor<1x4x32xf32>) -> tensor<1x4x32x1xf32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2, 3] : (tensor<1x4x32x1xf32>) -> tensor<1x4x32x32xf32>
    %231 = stablehlo.subtract %227, %230 : tensor<1x4x32x32xf32>
    %232 = stablehlo.exponential %231 : tensor<1x4x32x32xf32>
    %233 = stablehlo.reduce(%232 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x4x32x32xf32>, tensor<f32>) -> tensor<1x4x32xf32>
    %234 = stablehlo.reshape %233 : (tensor<1x4x32xf32>) -> tensor<1x4x32x1xf32>
    %235 = stablehlo.broadcast_in_dim %234, dims = [0, 1, 2, 3] : (tensor<1x4x32x1xf32>) -> tensor<1x4x32x32xf32>
    %236 = stablehlo.divide %232, %235 : tensor<1x4x32x32xf32>
    %237 = stablehlo.reshape %236 : (tensor<1x4x32x32xf32>) -> tensor<4x32x32xf32>
    %238 = stablehlo.reshape %218 : (tensor<1x4x32x32xf32>) -> tensor<4x32x32xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2] : (tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %240 = stablehlo.dot_general %237, %239, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x32x32xf32>, tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %241 = stablehlo.reshape %240 : (tensor<4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %242 = stablehlo.transpose %241, dims = [0, 2, 1, 3] : (tensor<1x4x32x32xf32>) -> tensor<1x32x4x32xf32>
    %243 = stablehlo.reshape %242 : (tensor<1x32x4x32xf32>) -> tensor<1x32x128xf32>
    %244 = stablehlo.transpose %arg10, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %245 = stablehlo.reshape %243 : (tensor<1x32x128xf32>) -> tensor<32x128xf32>
    %246 = stablehlo.dot_general %245, %244, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x128xf32>) -> tensor<32x128xf32>
    %247 = stablehlo.reshape %246 : (tensor<32x128xf32>) -> tensor<1x32x128xf32>
    %248 = stablehlo.add %160, %247 : tensor<1x32x128xf32>
    %249 = stablehlo.power %248, %4 : tensor<1x32x128xf32>
    %250 = stablehlo.reduce(%249 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x32x128xf32>, tensor<f32>) -> tensor<1x32xf32>
    %251 = stablehlo.reshape %250 : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %252 = stablehlo.divide %251, %10 : tensor<1x32x1xf32>
    %253 = stablehlo.add %252, %14 : tensor<1x32x1xf32>
    %254 = stablehlo.rsqrt %253 : tensor<1x32x1xf32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<1x32x1xf32>) -> tensor<1x32x128xf32>
    %256 = stablehlo.multiply %248, %255 : tensor<1x32x128xf32>
    %257 = stablehlo.transpose %arg11, dims = [1, 0] : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %258 = stablehlo.reshape %256 : (tensor<1x32x128xf32>) -> tensor<32x128xf32>
    %259 = stablehlo.dot_general %258, %257, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x512xf32>) -> tensor<32x512xf32>
    %260 = stablehlo.reshape %259 : (tensor<32x512xf32>) -> tensor<1x32x512xf32>
    %261 = stablehlo.maximum %260, %cst_10 : tensor<1x32x512xf32>
    %262 = stablehlo.power %261, %154 : tensor<1x32x512xf32>
    %263 = stablehlo.transpose %arg12, dims = [1, 0] : (tensor<128x512xf32>) -> tensor<512x128xf32>
    %264 = stablehlo.reshape %262 : (tensor<1x32x512xf32>) -> tensor<32x512xf32>
    %265 = stablehlo.dot_general %264, %263, contracting_dims = [1] x [0] : (tensor<32x512xf32>, tensor<512x128xf32>) -> tensor<32x128xf32>
    %266 = stablehlo.reshape %265 : (tensor<32x128xf32>) -> tensor<1x32x128xf32>
    %267 = stablehlo.add %248, %266 : tensor<1x32x128xf32>
    %268 = stablehlo.power %267, %4 : tensor<1x32x128xf32>
    %269 = stablehlo.reduce(%268 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x32x128xf32>, tensor<f32>) -> tensor<1x32xf32>
    %270 = stablehlo.reshape %269 : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %271 = stablehlo.divide %270, %10 : tensor<1x32x1xf32>
    %272 = stablehlo.add %271, %14 : tensor<1x32x1xf32>
    %273 = stablehlo.rsqrt %272 : tensor<1x32x1xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2] : (tensor<1x32x1xf32>) -> tensor<1x32x128xf32>
    %275 = stablehlo.multiply %267, %274 : tensor<1x32x128xf32>
    %276 = stablehlo.transpose %arg13, dims = [1, 0] : (tensor<65536x128xf32>) -> tensor<128x65536xf32>
    %277 = stablehlo.reshape %275 : (tensor<1x32x128xf32>) -> tensor<32x128xf32>
    %278 = stablehlo.dot_general %277, %276, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x65536xf32>) -> tensor<32x65536xf32>
    %279 = stablehlo.reshape %278 : (tensor<32x65536xf32>) -> tensor<1x32x65536xf32>
    %280 = stablehlo.convert %cst_16 : (tensor<1xi64>) -> tensor<1xf32>
    %281 = stablehlo.reshape %280 : (tensor<1xf32>) -> tensor<f32>
    %282 = stablehlo.broadcast_in_dim %281, dims = [] : (tensor<f32>) -> tensor<1x32x65536xf32>
    %283 = stablehlo.divide %279, %282 : tensor<1x32x65536xf32>
    %284 = stablehlo.tanh %283 : tensor<1x32x65536xf32>
    %285 = stablehlo.multiply %284, %282 : tensor<1x32x65536xf32>
    %286 = stablehlo.reshape %285 : (tensor<1x32x65536xf32>) -> tensor<32x65536xf32>
    %287 = stablehlo.reshape %arg15 : (tensor<1x32xi64>) -> tensor<32xi64>
    %288 = stablehlo.reduce(%286 init: %cst_6) applies stablehlo.maximum across dimensions = [1] : (tensor<32x65536xf32>, tensor<f32>) -> tensor<32xf32>
    %289 = stablehlo.reshape %288 : (tensor<32xf32>) -> tensor<32x1xf32>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x65536xf32>
    %291 = stablehlo.subtract %286, %290 : tensor<32x65536xf32>
    %292 = stablehlo.exponential %291 : tensor<32x65536xf32>
    %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<32x65536xf32>, tensor<f32>) -> tensor<32xf32>
    %294 = stablehlo.reshape %293 : (tensor<32xf32>) -> tensor<32x1xf32>
    %295 = stablehlo.log %294 : tensor<32x1xf32>
    %296 = stablehlo.broadcast_in_dim %295, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x65536xf32>
    %297 = stablehlo.subtract %291, %296 : tensor<32x65536xf32>
    %298 = stablehlo.compare  NE, %287, %c,  SIGNED : (tensor<32xi64>, tensor<32xi64>) -> tensor<32xi1>
    %299 = stablehlo.broadcast_in_dim %298, dims = [0] : (tensor<32xi1>) -> tensor<32xi1>
    %300 = stablehlo.broadcast_in_dim %287, dims = [0] : (tensor<32xi64>) -> tensor<32xi64>
    %301 = stablehlo.select %299, %300, %c_1 : tensor<32xi1>, tensor<32xi64>
    %302 = stablehlo.reshape %301 : (tensor<32xi64>) -> tensor<32x1xi64>
    %303 = stablehlo.iota dim = 0 : tensor<32x1x1xi64>
    %304 = stablehlo.reshape %302 : (tensor<32x1xi64>) -> tensor<32x1x1xi64>
    %305 = stablehlo.concatenate %303, %304, dim = 2 : (tensor<32x1x1xi64>, tensor<32x1x1xi64>) -> tensor<32x1x2xi64>
    %306 = "stablehlo.gather"(%297, %305) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<32x65536xf32>, tensor<32x1x2xi64>) -> tensor<32x1xf32>
    %307 = stablehlo.reshape %306 : (tensor<32x1xf32>) -> tensor<32xf32>
    %308 = stablehlo.negate %307 : tensor<32xf32>
    %309 = stablehlo.broadcast_in_dim %308, dims = [0] : (tensor<32xf32>) -> tensor<32xf32>
    %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %311 = stablehlo.select %299, %309, %310 : tensor<32xi1>, tensor<32xf32>
    %312 = stablehlo.convert %298 : (tensor<32xi1>) -> tensor<32xi64>
    %313 = stablehlo.reduce(%312 init: %c_5) applies stablehlo.add across dimensions = [0] : (tensor<32xi64>, tensor<i64>) -> tensor<i64>
    %314 = stablehlo.convert %313 : (tensor<i64>) -> tensor<f32>
    %315 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %316 = stablehlo.divide %315, %314 : tensor<f32>
    return %316 : tensor<f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_1_32_1_16_torch.bfloat16_1: "0x020000000000000000000000000000000000000000000000000000000000000000000000573F083F9F3E353ECC3D663D023D923C243CB83B4F3BE93A833A133AA6393A39693F673F173FB23E4B3EE63D813D123DA43C383CCF3B693B033B933A263ABA39113E7E3F503F023F973E2C3EC23D5A3DF63C8A3C1B3CAF3B453BDD3A793A0C3A42BF473F743F273FC73E643E013E923D243DB83C4F3CE93B833B133BA63A3A3A75BFA63E803F473FF53E8E3E213EB63D4D3DE63C823C123CA43B383BCF3A693A8FBE6CBE723F603F113FA93E413EDA3D763D0A3D9B3C2F3CC53B5D3BF93A8C3A283F37BF4D3F733F253FC43E613EFE3D8F3D213DB53C4C3CE53B813B113BA33A7D3F7ABF133F7D3F383FDF3E803E113EA43D383DCF3C693C033C933B263BBA3AD33E71BF953E803F493FF83E903E233EB83D4F3DE93C833C133CA63B3B3BD23A0BBF1DBFA9BC7B3F573F083F9F3E353ECC3D663D023D923C243CB83B4F3BE93A80BFC7BDA9BE6D3F643F143FAF3E473EE13D7D3D0E3DA03C343CCB3B643B003B09BFE63E1CBF583F6F3F203FBE3E593EF53D8A3D1B3DAF3C453CDD3B793B0C3BD73E5B3F53BF3D3F773F2B3FCD3E6B3E053E963D283DBD3C553CF03B873B183B7E3F803F76BF1B3F7C3F353FDB3E7C3E0F3EA13D353DCC3C653C013C913B233B263F563F80BFEA3E7F3F3F3FEA3E873E193EAD3D423DDA3C763C0A3C9B3B2F3B93BED43E71BF963E803F483FF83E903E233EB83D4F3DE93C833C133CA63B3A3B76BF0ABE4ABFF23D7E3F513F033F983E2D3EC33D5C3DF83C8B3C1D3CB03B463B40BF24BF0FBF73BD793F593F0A3FA13E373ECF3D693D033D933C263CBB3B523B193E74BF8BBE71BE723F603F113FAA3E413EDA3D763D0A3D9C3C2F3CC53B5D3B6A3F78BF293DCEBE693F673F173FB23E4B3EE63D813D123DA43C383CCF3B693B563F30BFB33E0FBF5D3F6D3F1E3FBB3E553EF13D883D193DAC3C413CDA3B753B11BC46BE203F32BF4F3F723F243FC33E5F3EFD3D8E3D203DB43C4B3CE43B803B59BFB83E563F50BF3F3F763F2A3FCC3E693E043E953D273DBC3C543CEE3B863B68BF4D3F773F67BF2D3F7A3F303FD43E733E0A3E9B3D2F3DC53C5D3CF93B8C3B08BE7F3F803F77BF193F7D3F363FDC3E7D3E0F3EA23D363DCD3C663C023C923B433F633F6F3F7FBF043F7E3F3C3FE43E843E153EA83D3D3DD53C703C073C983B753F003F463F7FBFDB3E803F413FEC3E893E1B3EAF3D453DDD3C793C0C3C9D3B8B3E1ABD0A3F77BFAC3E803F463FF53E8D3E213EB53D4C3DE53C813C113CA33B2ABF11BF813E67BF753E803F4B3FFC3E923E263EBC3D533DEE3C863C163CA93B7DBF6BBF7EBD50BF113E7E3F503F023F973E2C3EC23D5A3DF63C8A3C1B3CAF3BCFBE7DBFBDBE32BF2A3D7C3F553F063F9C3E323EC83D623DFE3C8F3C213CB53B",
      torch_tensor_1_32_1_16_torch.bfloat16: "0x02000000803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F803F0A3F593F733F7C3F7F3F803F803F803F803F803F803F803F803F803F803F803FD5BEDD3E4E3F703F7B3F7E3F7F3F803F803F803F803F803F803F803F803F803F7DBFEDBD153F5C3F753F7C3F7F3F803F803F803F803F803F803F803F803F803F27BF21BF9A3E423F6C3F7A3F7E3F7F3F803F803F803F803F803F803F803F803F913E72BF29BC213F613F763F7D3F7F3F803F803F803F803F803F803F803F803F763F79BFA4BEF73E533F723F7B3F7F3F803F803F803F803F803F803F803F803F413F33BF19BFA43E443F6C3F7A3F7E3F7F3F803F803F803F803F803F803F803F15BE59BE52BF173E323F673F783F7D3F7F3F803F803F803F803F803F803F803F69BFAF3E75BFF3BC1F3F603F763F7D3F7F3F803F803F803F803F803F803F803F57BF4A3F80BF53BE0A3F593F733F7C3F7F3F803F803F803F803F803F803F803F913B7F3F72BFC0BEE83E513F713F7B3F7E3F803F803F803F803F803F803F803F583F653F4BBF09BFBA3E483F6E3F7A3F7E3F7F3F803F803F803F803F803F803F683F043F11BF2DBF893E3F3F6B3F793F7E3F7F3F803F803F803F803F803F803F0C3E9ABC90BE4BBF2E3E353F673F783F7D3F7F3F803F803F803F803F803F803F42BF0DBFFE3C64BF913D2A3F643F773F7D3F7F3F803F803F803F803F803F803F75BF69BFAE3E75BFEFBC1F3F603F763F7D3F7F3F803F803F803F803F803F803F8DBE7EBF1E3F7EBF04BE143F5C3F743F7C3F7F3F803F803F803F803F803F803F293F44BF553F80BF69BE083F583F733F7C3F7F3F803F803F803F803F803F803F7D3F9DBE763F79BFA6BEF73E533F723F7B3F7F3F803F803F803F803F803F803FD13E7F3E803F6ABFD5BEDD3E4E3F703F7B3F7E3F7F3F803F803F803F803F803F0CBF3A3F703F54BF01BFC33E4A3F6E3F7A3F7E3F7F3F803F803F803F803F803F80BF7B3F483F38BF17BFA83E453F6D3F7A3F7E3F7F3F803F803F803F803F803F08BF6F3F0C3F15BF2BBF8C3E3F3F6B3F793F7E3F7F3F803F803F803F803F803FD93E193F863EDCBE3DBF613E3A3F693F793F7E3F7F3F803F803F803F803F803F7E3FA13D54BD87BE4DBF283E343F673F783F7D3F7F3F803F803F803F803F803F263FEEBEB8BEB6BD5BBFDE3D2E3F653F773F7D3F7F3F803F803F803F803F803F96BE5EBF22BFB63D67BF573D283F633F773F7D3F7F3F803F803F803F803F803F76BF80BF57BF873E71BF76BB223F613F763F7D3F7F3F803F803F803F803F803F40BF53BF78BFDC3E79BF76BD1C3F5F3F753F7D3F7F3F803F803F803F803F803F1E3ECBBE80BF153F7DBFEDBD153F5C3F753F7C3F7F3F803F803F803F803F803F6A3F1D3E6EBF383F80BF30BE0F3F5A3F743F7C3F7F3F803F803F803F803F803F"
    }
  }
#-}
