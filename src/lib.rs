use core::ops::{Index, IndexMut};
use rand::thread_rng;
use rand::seq::SliceRandom;
use std::iter::{FusedIterator};
use torcher::tensor::{Tensor, TensorView};

pub fn time_step<T>(tensor: T, dim: usize, size: usize, step: usize) -> T where T: Tensor {
    unsafe {
        tensor.new_unfold(dim, size, step)
    }
}

pub enum IncompleteOp {
    /// Return an incomplete item as it is.
    /// The size will be different than the rest.
    AsIs,
    /// Don't return an incomplete item.
    Drop,
    /// Pad the incomplete item to make an item the same size
    /// as other
    Pad(Align)
}

/// When an item need padding, the content will need to be 
/// aligned to some anchor.
pub enum Align {
    /// Align content at center, if expected length of item is 2n + 1
    /// but current length of item is 2n, padding before the first element
    /// will have one less element. The same goes for expected length 
    /// of item is 2n but current length is 2n + 1, padding before the first
    /// element will have one less element.
    /// For example:
    /// If expected length is 5 like:
    /// [nnnnn]
    /// The data length is 2 will be align like:
    /// [nxxnn]
    /// If expected length is 4 but the data length is 1, it'll be align like:
    /// [nxnn]
    Center,
    /// The padding will be done after the last
    /// item
    Begin,
    /// The padding will be done before the first element.
    End
}

/// Group tensor into batch according to `batch_size`.
/// The last batch may be "incomplete" because there may not be
/// enough number of remaining element. 
/// User need to specify how `incomplete_op` to pick what operation
/// needed on the "incomplete" batch.
/// 
/// It take a closure/function that will be repeatly called to get
/// next tensor to be included in the batch. When the function
/// return None, that mean there's no more element to include.
/// 
/// When tensor being included into the batch, it'll a clone per each element in tensor.
/// The batched tensor will have continguous stride.
pub struct BatchTensor<F, P, T> 
where F: FnMut() -> Option<T>, 
      T: Tensor<Datum=P> {
    batch_size: usize,
    data_fn: F,
    incomplete_op: IncompleteOp,
    pad_value: P
}

impl<F, P, T> BatchTensor<F, P, T> 
where F: FnMut() -> Option<T>, 
      T: Tensor<Datum=P> {
    /// Create a new iterator that return batch of tensor
    pub fn new(incomplete: IncompleteOp, pad: P, size: usize, f: F) -> Self {
        Self {
            batch_size: size,
            data_fn: f,
            incomplete_op: incomplete,
            pad_value: pad,
        }
    }
}

impl<F, P, T> Iterator for BatchTensor<F, P, T> 
where F: FnMut() -> Option<T>, 
      T: Tensor<Datum=P>,
      P: Copy
{
    type Item = TensorView<T>;

    fn next(&mut self) -> Option<Self::Item> {
        fn fill_data<T: Tensor<Datum=P>, P: Copy>(sources: &[T], target_data: &mut [P]) {
            let mut i = 0;
            for source in sources {
                for elm in source.iter() {
                    target_data[i] = elm;
                    i += 1;
                }
            }
        }

        let mut buffer = Vec::with_capacity(self.batch_size);
        let mut i = 0;
        let mut shape = vec![];

        while let Some(t) = (self.data_fn)() {
            if shape.len() > 0 {
                if shape != t.shape().0 {
                    panic!("The tensor return from function have shape {:?} while prior tensor have shape {:?}.", shape, t.shape().0);
                }
            } else {
                shape = t.shape().0.to_owned();
            }

            buffer.push(t);
            i += 1;

            if i == self.batch_size {
                break;
            }
        }

        if shape.len() > 0 {
            let dim = shape.len() + 1;
            let mut new_shape = Vec::with_capacity(dim);
            new_shape.push(Some(i));
            // Most of the time, i == batch_size, except
            // last batch that may have different value
            let mut numel = i;
            shape.iter().for_each(|size| {
                new_shape.push(Some(*size));
                numel *= size;
            });

            if i != self.batch_size {
                // need pad
                match &self.incomplete_op {
                    IncompleteOp::AsIs => {
                        let mut batch = T::new_with_size_1d(numel);
                        fill_data(&buffer, batch.data_mut());
                        Some(batch.view(new_shape.as_slice()).unwrap())
                    },
                    IncompleteOp::Drop => {
                        None
                    },
                    IncompleteOp::Pad(align) => {
                        match align {
                            Align::Begin => {
                                let pad_start = numel;
                                // recalculate numel for expected batch_size
                                numel /= i;
                                numel *= self.batch_size;
                                let mut batch = T::new_with_size_1d(numel);
                                fill_data(&buffer, batch.data_mut());
                                // pad remaining elements
                                let pad_area = &mut batch.data_mut()[pad_start..];
                                pad_area.iter_mut().for_each(|p| *p = self.pad_value);
                                Some(batch.view(new_shape.as_slice()).unwrap())
                            },
                            Align::Center => {
                                // replace batch size with expected batch size
                                new_shape.remove(0);
                                new_shape.insert(0, Some(self.batch_size));
                                // number of actual data elements
                                let data_numel = numel;
                                // expected number of data elements
                                numel /= i;
                                numel *= self.batch_size;
                                // last pad element before data
                                let pad_end = (numel - data_numel) / 2;
                                // last data element 
                                let data_end = pad_end + data_numel;
                                let mut batch = T::new_with_size_1d(numel);
                                // fill data start after pad
                                fill_data(&buffer, &mut batch.data_mut()[pad_end..]);
                                // put pad before data element
                                let pad_area = &mut batch.data_mut()[..pad_end];
                                pad_area.iter_mut().for_each(|p| *p = self.pad_value);
                                // put pad after data element
                                let pad_area = &mut batch.data_mut()[data_end..];
                                pad_area.iter_mut().for_each(|p| *p = self.pad_value);
                                Some(batch.view(new_shape.as_slice()).unwrap())
                            },
                            Align::End => {
                                // replace batch size with expected batch size
                                new_shape.remove(0);
                                new_shape.insert(0, Some(self.batch_size));
                                // number of actual data elements
                                let data_numel = numel;
                                // expected number of data elements
                                numel /= i;
                                numel *= self.batch_size;
                                // last padded element
                                let pad_end = numel - data_numel;
                                let mut batch = T::new_with_size_1d(numel);
                                // fill data start after padded element
                                fill_data(&buffer, &mut batch.data_mut()[pad_end..]);
                                // put pad element
                                let pad_area = &mut batch.data_mut()[..pad_end];
                                pad_area.iter_mut().for_each(|p| *p = self.pad_value);
                                Some(batch.view(new_shape.as_slice()).unwrap())
                            }
                        }
                    }
                }
            } else {
                // Batch is filled with tensor. No need for padding
                let mut batch = T::new_with_size_1d(numel);
                fill_data(&buffer, batch.data_mut());
                Some(batch.view(new_shape.as_slice()).unwrap())
            }
        } else {
            // no single tensor return from data function
            None
        }
    }
}

#[derive(Debug)]
pub struct OutOfBoundErr;

/// A tensor iterator return from IndicedTensor
pub struct TensorIterator<T>
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{
    batch_numel: usize,
    current: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,

    batch_size: usize,
    drop_last: bool,
    tensor: TensorIteratorSrc<T>
}

enum TensorIteratorSrc<T>
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{
    Compat(IndicedTensor<T>),
    Incompat(IndicedTensorIterator<T>)
}

impl<T> TensorIterator<T> 
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{
    fn new(batch_size: usize, drop_last: bool, tensor: IndicedTensor<T>) -> TensorIterator<T> {
        let mut shape = tensor.tensor.shape().0.to_owned();
        // add batch size to first index of shape
        shape.insert(0, batch_size);
        let mut stride = Vec::with_capacity(shape.len());

        shape.iter().fold(1, |cum, s| {
            stride.push(cum);
            cum * s
        }); // stride ignore last cum * s

        if tensor.compat {
            // batch dim is now on last dim
            let mut batch = tensor.unfold(0, batch_size, batch_size).unwrap();

            // keep transpose batch dim until it is first dim without
            // changing other dim order
            for i in (1..shape.len()).rev() {
                batch = batch.transpose(i, i - 1).unwrap();
            }

            TensorIterator {
                batch_numel: shape.iter().product(),
                current: 0,
                shape: shape,
                stride: stride,

                batch_size: batch_size,
                drop_last: drop_last,
                tensor: TensorIteratorSrc::Compat(batch)
            }
        } else {
            TensorIterator {
                batch_numel: shape.iter().product(),
                current: 0,
                shape: shape,
                stride: stride,

                batch_size: batch_size,
                drop_last: drop_last,
                tensor: TensorIteratorSrc::Incompat(tensor.into_iter())
            }
        }
    }
}

impl<T> ExactSizeIterator for TensorIterator<T> 
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{
    fn len(&self) -> usize {
        self.shape[0]
    }
}

impl<T> FusedIterator for TensorIterator<T>
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{ }

impl<T> Iterator for TensorIterator<T> 
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let mut batch = T::new_with_size_1d(self.batch_numel);
        let data = batch.data_mut();
        match &mut self.tensor {
            TensorIteratorSrc::Compat(tensor) => {
                if self.current >= self.shape[0] {
                    None
                } else {
                    unsafe {
                        let result = tensor.tensor.new_narrow(0, self.current, self.batch_size);
                        self.current += self.batch_size;

                        Some(result)
                    }
                }
            },
            TensorIteratorSrc::Incompat(tensor) => {
                for i in 0..self.batch_size {
                    // filling batch
                    for j in 0..self.batch_numel {
                        // filling each of batch with data
                        match tensor.next() {
                            // more data available, copy to batch
                            Some(d) => data[i * self.batch_numel + j] = d,

                            // no more data available but batch is yet to be fill
                            None => {
                                if self.drop_last {
                                    // drop_last so ignore this last batch
                                    return None
                                } else if i > 0 {
                                    // i > 0 mean there's at least a batch succeed.
                                    // Return as is.
                                    // since all tensors have same shape.
                                    // Each tensor in batch shall be complete.
                                    // Only batch that's not fulfill
                                    
                                    // resize shape according to batching progress.
                                    self.shape.remove(0);
                                    self.shape.insert(0, i + 1);
                                    batch.resize_nd(&self.shape, &self.stride);
                                    return Some(batch);
                                } else {
                                    // no more batch available
                                    return None
                                }
                            }
                        }
                    }
                }
                
                // resize batch according to pre-calculated shape/stride
                batch.resize_nd(&self.shape, &self.stride);

                Some(batch)
            }
        }
    }
}

/// A Tensor representation that provide safety abstraction over original tensor.
/// Easiest way to construct this struct is from `IndicedTensor::from(tensor)`
/// where tensor is an object that implement trait `torcher::tensor::Tensor`.
/// 
/// Normal use-case for this struct is to make tensor ready to be feeded to 
/// "PyTorch" API.
/// The lower level use-case like algorithmic tensor generation or mathematic
/// operation to populate tensor shall use `torcher` crate.
pub struct IndicedTensor<T> where T: Tensor {
    compat: bool, // True is current object can be map directly to PyTorch
    dim: Vec<usize>, // A dim that need index map
    idx: Vec<Vec<usize>>, // An index map. 
    origin: T,
    tensor: T
}

/// Provide most common function that usually done on tensor when
/// feeding into any kind of model.
/// This is a safety abstraction that shield user from unsafe code.
impl<T> IndicedTensor<T>
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy {
    
    /// Finalize the IndicedTensor by consume it and return a batch iterator
    /// that on each iteration return a tensor.
    /// There will be a copy op on each data element if there's at least one 
    /// operation done on this object that's incompatible with PyTorch.
    /// 
    /// For example, [shuffled](struct.IndicedTensor.html#method.shuffle) method
    pub fn as_batch(self, batch_size: usize, drop_last: bool) -> TensorIterator<T> {
        TensorIterator::new(batch_size, drop_last, self)
    }

    /// Transform this object into an actual tensor.
    /// This may create a new tensor that disconnect itself
    /// from the original tensor if some operation applied to this
    /// object have no similar functionality (Incompatible) on PyTorch's tensor
    /// counterpart. The most notable one here is 
    /// [shuffled](struct.IndicedTensor.html#method.shuffle) tensor.
    /// This method return the original tensor along
    /// with operated tensor. If no incompatible operation
    /// is applied, it'll reuse the storage of original tensor.
    /// It's safe to assume that original tensor need to live as long
    /// as the operated tensor.
    pub fn as_tensor(self) -> (T, T) {
        if self.compat {
            (self.origin, self.tensor)
        } else {
            // incompatible op so it need to create new independent tensor
            let numel = self.tensor.numel();
            let mut tensor = T::new_with_size_1d(numel);
            let (shape, _) = self.tensor.shape();
            let mut stride = Vec::with_capacity(shape.len());
            shape.iter().fold(1, |cum, s| {
                stride.insert(0, cum);
                cum * s
            });
            tensor.resize_nd(shape, &stride);
            let data = tensor.data_mut();

            // the iterator of self will honor all applied op so
            // we use it to ensure that all op are honor
            self.iter().enumerate().for_each(|(i, v)| data[i] = v); // copy op

            (self.origin, tensor)
        }
    }
    
    pub fn iter(&self) -> BorrowIndicedTensorIterator<T> {
        self.into_iter()
    }

    /// Shuffle tensor on given dim. This doesn't actually permute tensor.
    /// It'll just create a map that will be used on access.
    /// The rationale behind this is to make randomized tensor reusable on
    /// more use-cases.
    /// For example, batch of randomized set of temporal data to be feeded to RNN.
    /// 
    /// This operation make it incompatible with PyTorch.
    /// This mean that [as_tensor](struct.IndicedTensor.html#method.as_tensor) and
    /// [as_batch](struct.IndicedTensor.html#method.as_batch) 
    /// will return deep copy of tensor.
    pub fn shuffle(mut self, dim: usize) -> Result<IndicedTensor<T>, OutOfBoundErr> {
        let shape = self.tensor.shape().0;
        if dim < shape.len() {
            let i = match self.dim.binary_search(&dim) {
                Ok(_) => return Ok(self),
                Err(i) => i
            };
            self.dim.insert(i, dim);
            let mut random = Vec::with_capacity(shape[dim]);
            (0..shape[dim]).for_each(|v| random.push(v));
            let mut rng = thread_rng();
            random.shuffle(&mut rng);
            self.idx.insert(i, random);

            Ok(self)
        } else {
            Err(OutOfBoundErr)
        }
    }

    /// Transpose the tensor by using size/stride trick. This doesn't actually 
    /// swap any element. This operation is cheap but it make tensor
    /// incontiguous.
    /// 
    /// This operation doesn't effect PyTorch compatibility
    pub fn transpose(mut self, dim_1: usize, dim_2: usize) -> Result<IndicedTensor<T>, OutOfBoundErr> {
        unsafe {
            let tensor = self.tensor.new_transpose(dim_1, dim_2);
            self.tensor = tensor;
            let mut rand_1 = None;
            let mut rand_2 = None;

            match self.dim.binary_search(&dim_1) {
                Ok(i) => {
                    self.dim.remove(i);
                    rand_2 = Some(()); // shuffle dim_2 later because dim_1 become dim_2
                },
                Err(_) => ()
            }

            match self.dim.binary_search(&dim_2) {
                Ok(i) => {
                    self.dim.remove(i);
                    rand_1 = Some(()); // shuffle dim_1 later because dim_2 become dim_1
                }
                Err(_) => ()
            }

            let mut cur_tensor = self;

            if rand_1.is_some() {
                cur_tensor = cur_tensor.shuffle(dim_1)?;
            }

            if rand_2.is_some() {
                cur_tensor = cur_tensor.shuffle(dim_2)?;
            }

            Ok(cur_tensor)
        }
    }
    
    /// Unfold given dim into new dim. The new dim will be put on last dim.
    /// The dim that got unfold will be shrink per given size and step.
    /// 
    /// If the given dim is already random, the random will be re-random due
    /// to the change of dim size. The new dim will be ordered according to
    /// original order of data.
    /// 
    /// For example, if tensor is:
    /// `[1, 2, 3, 4, 5, 6]`
    /// 
    /// After unfold(0, 3, 1):
    /// `[[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]`
    /// 
    /// After unfold(0, 3, 2):
    /// `[[1, 2, 3], [3, 4, 5]]`
    /// 
    /// If the first tensor is random then unfold like random(0).unfold(0, 3, 1) may get:
    /// `[[2, 3, 4], [1, 2, 3], [3, 4, 5], [4, 5, 6]]`
    /// Notice that the order of dim 1 is still ordered.
    pub fn unfold(mut self, dim: usize, size: usize, step: usize) -> Result<IndicedTensor<T>, OutOfBoundErr> {
        fn make_tensor<T>(compat: bool, ts: IndicedTensor<T>, unfolded: T) -> IndicedTensor<T> where T: Tensor{
            IndicedTensor {
                compat: compat,
                dim: ts.dim,
                idx: ts.idx,
                origin: ts.origin,
                tensor: unfolded
            }
        }

        if dim >= self.tensor.shape().0.len() {
            return Err(OutOfBoundErr);
        }

        unsafe {
            let unfolded = self.tensor.new_unfold(dim, size, step);

            match self.dim.binary_search(&dim) {
                Ok(i) => {
                    // Shape on ramdom dim is changed.
                    // It need to be re-random again
                    self.dim.remove(i);
                    self.idx.remove(i);

                    make_tensor(false, self, unfolded).shuffle(i)
                },
                Err(_) => Ok(make_tensor(true, self, unfolded))
            }

        }
    }
}

impl<T> IntoIterator for IndicedTensor<T> 
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{
    type Item = <T as Tensor>::Datum;
    type IntoIter = IndicedTensorIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        IndicedTensorIterator {
            cur_idx: self.tensor.shape().0.iter().map(|_| 0).collect(),
            tensor: self
        }
    }
}

impl<'a, T> IntoIterator for &'a IndicedTensor<T> 
where for<'b> T: Index<&'b [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{
    type Item = <T as Tensor>::Datum;
    type IntoIter = BorrowIndicedTensorIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        BorrowIndicedTensorIterator {
            cur_idx: self.tensor.shape().0.iter().map(|_| 0).collect(),
            tensor: self
        }
    }
}

pub struct IndicedTensorIterator<T> where T: Tensor {
    cur_idx: Vec<usize>,
    tensor: IndicedTensor<T>
}

/// Utility function to calculate next tensor index.
/// 
/// We need this indirection because *IndicedTensorIterator doesn't
/// share common trait except Iterator trait itself which we cannot use
/// as it doesn't provide necessary information to get actual tensor data.
/// 
/// The caller is responsible to map the index to expected index.
/// For example, shuffled index map.
fn _indiced_tensor_iterator_next_idx(cur_idx: &mut [usize], shape: &[usize])
{
    let last_idx = cur_idx.len();

    for i in (0..last_idx).rev() {
        cur_idx[i] += 1;

        if cur_idx[i] < shape[i] {
            break;
        } else if i != 0 {
            cur_idx[i] = 0;
        }
    }
}

impl<T> ExactSizeIterator for IndicedTensorIterator<T> 
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{
    fn len(&self) -> usize {
        self.tensor.tensor.numel()
    }
}

impl<T> FusedIterator for IndicedTensorIterator<T> 
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{}

impl<T> Iterator for IndicedTensorIterator<T> 
where for<'a> T: Index<&'a [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{
    type Item = <T as Tensor>::Datum;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_idx[0] < self.tensor.tensor.shape().0[0] {
            let result = Some(self.tensor[&self.cur_idx]);
            _indiced_tensor_iterator_next_idx(&mut self.cur_idx, &self.tensor.tensor.shape().0);
            result
        } else {
            None
        }
    }
}

pub struct BorrowIndicedTensorIterator<'a, T> where T: Tensor {
    cur_idx: Vec<usize>,
    tensor: &'a IndicedTensor<T>
}

impl<'a, T> Iterator for BorrowIndicedTensorIterator<'a, T> 
where for<'b> T: Index<&'b [usize], Output=<T as Tensor>::Datum> + Tensor,
      <T as Tensor>::Datum: Copy
{
    type Item = <T as Tensor>::Datum;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_idx[0] < self.tensor.tensor.shape().0[0] {
            let result = Some(self.tensor[&self.cur_idx]);
            _indiced_tensor_iterator_next_idx(&mut self.cur_idx, &self.tensor.tensor.shape().0);
            result
        } else {
            None
        }
    }
}

impl<T> From<T> for IndicedTensor<T> where T: Tensor {
    fn from(tensor: T) -> Self {
        unsafe {
            // perform cheap shallow clone
            let cloned = tensor.new_narrow(0, 0, tensor.shape().0[0]);

            IndicedTensor {
                compat: true,
                dim: Vec::new(),
                idx: Vec::new(),
                origin: tensor,
                tensor: cloned
            }
        }
    }
}

impl<T> Index<&[usize]> for IndicedTensor<T> where T: Tensor {
    type Output = <T as Tensor>::Datum;

    fn index(&self, idx: &[usize]) -> &Self::Output {
        let (_, stride) = self.tensor.shape();
        
        let offset = idx.iter().enumerate().fold(0, |cum, (dim, i)| {
            match self.dim.iter().enumerate().find(|d| *d.1 == dim) {
                Some(d) => {
                    cum + (self.idx[d.0][*i] * stride[dim])
                },
                None => cum + *i * stride[dim]
            }
            
        });

        &self.tensor.data()[offset]
    }
}

impl<T> IndexMut<&[usize]> for IndicedTensor<T> where T: Tensor {
    fn index_mut(&mut self, idx: &[usize]) -> &mut Self::Output {
        let stride = self.tensor.shape().1;
        let adjusted_idx = idx.iter().enumerate().fold(0, |cum, (dim, i)| {
            match self.dim.iter().enumerate().find(|d| *d.1 == dim) {
                Some(d) => cum + (self.idx[d.0][*i] * stride[dim]),
                None => cum
            }
            
        });

        &mut self.tensor.data_mut()[adjusted_idx]
    }
}
#[cfg(test)]
mod tests;