module {
  func.func @main(%arg0: tensor<65x32xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<65x32xf32>, %arg5: tensor<65xf32>, %arg6: tensor<1x8xi64>, %arg7: tensor<1x8xi64>) -> tensor<f32> {
    %c = stablehlo.constant dense<-100> : tensor<8xi64>
    %c_0 = stablehlo.constant dense<0> : tensor<8xi64>
    %c_1 = stablehlo.constant dense<1> : tensor<8xi64>
    %c_2 = stablehlo.constant dense<8> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_4 = stablehlo.constant dense<0> : tensor<i64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_6 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_7 = arith.constant dense<1> : tensor<1xi64>
    %cst_8 = arith.constant dense<32> : tensor<1xi64>
    %cst_9 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %0 = "stablehlo.gather"(%arg0, %arg6) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<65x32xf32>, tensor<1x8xi64>) -> tensor<1x8x32xf32>
    %1 = stablehlo.convert %0 : tensor<1x8x32xf32>
    %2 = stablehlo.convert %c_3 : (tensor<i64>) -> tensor<f64>
    %3 = stablehlo.convert %c_2 : (tensor<i64>) -> tensor<f64>
    %4 = stablehlo.divide %3, %2 : tensor<f64>
    %5 = stablehlo.ceil %4 : tensor<f64>
    %6 = stablehlo.convert %5 : (tensor<f64>) -> tensor<i64>
    %7 = stablehlo.reshape %6 : (tensor<i64>) -> tensor<1xi64>
    %8 = stablehlo.dynamic_iota %7, dim = 0 : (tensor<1xi64>) -> tensor<8xi64>
    %9 = stablehlo.multiply %8, %c_1 : tensor<8xi64>
    %10 = stablehlo.add %9, %c_0 : tensor<8xi64>
    %11 = "stablehlo.gather"(%arg1, %10) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<8x32xf32>, tensor<8xi64>) -> tensor<8x32xf32>
    %12 = stablehlo.convert %11 : tensor<8x32xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [1, 2] : (tensor<8x32xf32>) -> tensor<1x8x32xf32>
    %14 = stablehlo.add %1, %13 : tensor<1x8x32xf32>
    %15 = stablehlo.convert %14 : (tensor<1x8x32xf32>) -> tensor<1x8x32xf64>
    %16 = stablehlo.reduce(%15 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<1x8x32xf64>, tensor<f64>) -> tensor<1x8xf64>
    %17 = stablehlo.reshape %16 : (tensor<1x8xf64>) -> tensor<1x8x1xf64>
    %18 = stablehlo.convert %cst_8 : (tensor<1xi64>) -> tensor<1xf64>
    %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f64>) -> tensor<1x8x1xf64>
    %21 = stablehlo.divide %17, %20 : tensor<1x8x1xf64>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<1x8x1xf64>) -> tensor<1x8x32xf64>
    %23 = stablehlo.subtract %15, %22 : tensor<1x8x32xf64>
    %24 = stablehlo.multiply %23, %23 : tensor<1x8x32xf64>
    %25 = stablehlo.reduce(%24 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<1x8x32xf64>, tensor<f64>) -> tensor<1x8xf64>
    %26 = stablehlo.reshape %25 : (tensor<1x8xf64>) -> tensor<1x8x1xf64>
    %27 = stablehlo.divide %26, %20 : tensor<1x8x1xf64>
    %28 = stablehlo.convert %27 : (tensor<1x8x1xf64>) -> tensor<1x8x1xf32>
    %29 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x8x32xf32>, tensor<f32>) -> tensor<1x8xf32>
    %30 = stablehlo.reshape %29 : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %31 = stablehlo.convert %cst_8 : (tensor<1xi64>) -> tensor<1xf32>
    %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %34 = stablehlo.divide %30, %33 : tensor<1x8x1xf32>
    %35 = stablehlo.convert %cst_9 : (tensor<1xf64>) -> tensor<1xf32>
    %36 = stablehlo.reshape %35 : (tensor<1xf32>) -> tensor<f32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %38 = stablehlo.add %28, %37 : tensor<1x8x1xf32>
    %39 = stablehlo.rsqrt %38 : tensor<1x8x1xf32>
    %40 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x32xf32>
    %41 = stablehlo.subtract %14, %40 : tensor<1x8x32xf32>
    %42 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x32xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<1x8x32xf32>
    %44 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<1x8x32xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<1x8x32xf32>
    %46 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<32xf32>) -> tensor<1x8x32xf32>
    %47 = stablehlo.add %45, %46 : tensor<1x8x32xf32>
    %48 = stablehlo.reshape %47 : (tensor<1x8x32xf32>) -> tensor<8x32xf32>
    %49 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<65x32xf32>) -> tensor<32x65xf32>
    %50 = stablehlo.dot_general %48, %49, contracting_dims = [1] x [0] : (tensor<8x32xf32>, tensor<32x65xf32>) -> tensor<8x65xf32>
    %51 = stablehlo.convert %cst_7 : (tensor<1xi64>) -> tensor<1xf32>
    %52 = stablehlo.reshape %51 : (tensor<1xf32>) -> tensor<f32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [] : (tensor<f32>) -> tensor<8x65xf32>
    %54 = stablehlo.multiply %50, %53 : tensor<8x65xf32>
    %55 = stablehlo.broadcast_in_dim %52, dims = [] : (tensor<f32>) -> tensor<65xf32>
    %56 = stablehlo.multiply %arg5, %55 : tensor<65xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [1] : (tensor<65xf32>) -> tensor<8x65xf32>
    %58 = stablehlo.add %54, %57 : tensor<8x65xf32>
    %59 = stablehlo.reshape %58 : (tensor<8x65xf32>) -> tensor<1x8x65xf32>
    %60 = stablehlo.reshape %59 : (tensor<1x8x65xf32>) -> tensor<8x65xf32>
    %61 = stablehlo.reshape %arg7 : (tensor<1x8xi64>) -> tensor<8xi64>
    %62 = stablehlo.reduce(%60 init: %cst_6) applies stablehlo.maximum across dimensions = [1] : (tensor<8x65xf32>, tensor<f32>) -> tensor<8xf32>
    %63 = stablehlo.reshape %62 : (tensor<8xf32>) -> tensor<8x1xf32>
    %64 = stablehlo.broadcast_in_dim %63, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x65xf32>
    %65 = stablehlo.subtract %60, %64 : tensor<8x65xf32>
    %66 = stablehlo.exponential %65 : tensor<8x65xf32>
    %67 = stablehlo.reduce(%66 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x65xf32>, tensor<f32>) -> tensor<8xf32>
    %68 = stablehlo.reshape %67 : (tensor<8xf32>) -> tensor<8x1xf32>
    %69 = stablehlo.log %68 : tensor<8x1xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x65xf32>
    %71 = stablehlo.subtract %65, %70 : tensor<8x65xf32>
    %72 = stablehlo.compare  NE, %61, %c,  SIGNED : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0] : (tensor<8xi1>) -> tensor<8xi1>
    %74 = stablehlo.broadcast_in_dim %61, dims = [0] : (tensor<8xi64>) -> tensor<8xi64>
    %75 = stablehlo.select %73, %74, %c_0 : tensor<8xi1>, tensor<8xi64>
    %76 = stablehlo.reshape %75 : (tensor<8xi64>) -> tensor<8x1xi64>
    %77 = stablehlo.iota dim = 0 : tensor<8x1x1xi64>
    %78 = stablehlo.reshape %76 : (tensor<8x1xi64>) -> tensor<8x1x1xi64>
    %79 = stablehlo.concatenate %77, %78, dim = 2 : (tensor<8x1x1xi64>, tensor<8x1x1xi64>) -> tensor<8x1x2xi64>
    %80 = "stablehlo.gather"(%71, %79) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<8x65xf32>, tensor<8x1x2xi64>) -> tensor<8x1xf32>
    %81 = stablehlo.reshape %80 : (tensor<8x1xf32>) -> tensor<8xf32>
    %82 = stablehlo.negate %81 : tensor<8xf32>
    %83 = stablehlo.broadcast_in_dim %82, dims = [0] : (tensor<8xf32>) -> tensor<8xf32>
    %84 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %85 = stablehlo.select %73, %83, %84 : tensor<8xi1>, tensor<8xf32>
    %86 = stablehlo.convert %72 : (tensor<8xi1>) -> tensor<8xi64>
    %87 = stablehlo.reduce(%86 init: %c_4) applies stablehlo.add across dimensions = [0] : (tensor<8xi64>, tensor<i64>) -> tensor<i64>
    %88 = stablehlo.convert %87 : (tensor<i64>) -> tensor<f32>
    %89 = stablehlo.reduce(%85 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
    %90 = stablehlo.divide %89, %88 : tensor<f32>
    return %90 : tensor<f32>
  }
}
