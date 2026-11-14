#ifndef FUSION_IO_SRC_QUANTITY_METADATA_HPP
#define FUSION_IO_SRC_QUANTITY_METADATA_HPP

#include "fusion_io/fusion_io.hpp"

namespace fio::detail
{

/// Sets the units and properties of `quantity`'s values on `field`, as
/// Fusion-IO 1.x gives them: SI, temperatures in eV. Units that are not
/// certain stay unspecified.
void SetQuantityMetadata(pcms::Field<Real>& field, Quantity quantity);

} // namespace fio::detail

#endif // FUSION_IO_SRC_QUANTITY_METADATA_HPP
