import hypothesis._strategies as st
import hypothesis.internal.conjecture.utils as cu


class ValueIndexStrategy(st.SearchStrategy):
    def __init__(self, elements, dtype, min_size, max_size, unique):
        super(ValueIndexStrategy, self).__init__()
        self.elements = elements
        self.dtype = dtype
        self.min_size = min_size
        self.max_size = max_size
        self.unique = unique

    def do_draw(self, data):
        result = []
        seen = set()

        iterator = cu.many(
            data,
            min_size=self.min_size,
            max_size=self.max_size,
            average_size=(self.min_size + self.max_size) / 2,
        )

        while iterator.more():
            elt = data.draw(self.elements)

            if self.unique:
                if elt in seen:
                    iterator.reject()
                    continue
                seen.add(elt)
            result.append(elt)

        dtype = infer_dtype_if_necessary(
            dtype=self.dtype, values=result, elements=self.elements, draw=data.draw
        )
        return pandas.Index(result, dtype=dtype, tupleize_cols=False)