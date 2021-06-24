#pragma once

#include <quda_internal.h>
#include <quda.h>
#include <lattice_field.h>

#include <comm_key.h>

namespace quda {

  namespace misc
  {

    inline bool isNative(QudaMiscFieldOrder order, QudaPrecision precision)
    {
      if (precision == QUDA_DOUBLE_PRECISION) {
        if (order == QUDA_FLOAT2_FIELD_ORDER) return true;
      } else {
        if (order == QUDA_FLOAT4_FIELD_ORDER) return true;
      }
      return false;
    }

  } // namespace misc

  struct MiscFieldParam : public LatticeFieldParam {

    QudaFieldLocation location; // where are we storing the field (CUDA or GPU)?
    int nRow;
    int nCol;

    QudaMiscFieldOrder order;
    QudaTboundary t_boundary;

    void *misc; // used when we use a reference to an external field

    QudaFieldCreate create; // used to determine the type of field created

    QudaFieldGeometry geometry; // whether the field is a scale, vector or tensor

    // Default constructor
    MiscFieldParam(void *const h_misc = NULL) :
      LatticeFieldParam(),
      location(QUDA_INVALID_FIELD_LOCATION),
      nRow(1),
      nCol(1),
      nFace(0),
      order(QUDA_INVALID_FIELD_ORDER),
      t_boundary(QUDA_INVALID_T_BOUNDARY),
      misc(h_misc),
      create(QUDA_REFERENCE_FIELD_CREATE),
      geometry(QUDA_SCALAR_GEOMETRY),
      site_offset(0),
      site_size(0)
    {
    }

    MiscFieldParam(const MiscField &m);

    MiscFieldParam(const int *x, const QudaPrecision precision, const int pad,
                   const QudaFieldGeometry geometry, const QudaGhostExchange ghostExchange = QUDA_GHOST_EXCHANGE_PAD) :
      LatticeFieldParam(1, x, pad, precision, ghostExchange),
      location(QUDA_INVALID_FIELD_LOCATION),
      nRow(1),
      nCol(1),
      nFace(0),
      order(QUDA_INVALID_FIELD_ORDER),
      t_boundary(QUDA_INVALID_T_BOUNDARY),
      misc(0),
      create(QUDA_NULL_FIELD_CREATE),
      geometry(geometry),
      site_offset(0),
      site_size(0)
    {
    }

    MiscFieldParam(const QudaMiscParam &param, void *h_misc = nullptr) :
      LatticeFieldParam(param),
      location(QUDA_CPU_FIELD_LOCATION),
      nRow(1),
      nCol(1),
      nFace(0),
      order(param.misc_order),
      t_boundary(param.t_boundary),
      misc(h_misc),
      create(QUDA_REFERENCE_FIELD_CREATE),
      geometry(QUDA_SCALAR_GEOMETRY),
      site_offset(param.misc_offset),
      site_size(param.site_size)
    {
    }


    /**
       @brief Helper function for setting the precision and corresponding
       field order for QUDA internal fields.
       @param precision The precision to use
    */
    void setPrecision(QudaPrecision precision, bool force_native = false)
    {
      // is the current status in native field order?
      bool native = force_native ? true : misc::isNative(order, this->precision);
      this->precision = precision;
      this->ghost_precision = precision;

      if (native) {
        if (precision == QUDA_DOUBLE_PRECISION) {
          order = QUDA_FLOAT2_FIELD_ORDER;
        } else {
          order = QUDA_FLOAT4_FIELD_ORDER;
        }
      }
    }
  };

  std::ostream& operator<<(std::ostream& output, const MiscFieldParam& param);

  /// TODO TODO FIXME FIXME FIXME

  class MiscField : public LatticeField {

  protected:
      size_t bytes;        // bytes allocated per full field
      size_t length;
      size_t real_length;
      int nRow;
      int nCol;
      int nFace;
      QudaFieldGeometry geometry; // whether the field is a scale, vector or tensor

      QudaReconstructType reconstruct;
      int nInternal; // number of degrees of freedom per link matrix
      QudaMiscFieldOrder order;
      QudaTboundary t_boundary;

      QudaFieldCreate create; // used to determine the type of field created

      mutable void *ghost[2 * QUDA_MAX_DIM]; // stores the ghost zone of the misc field (non-native fields only)

      mutable int ghostFace[QUDA_MAX_DIM]; // the size of each face

      /**
         @brief Exchange the buffers across all dimensions in a given direction
         @param[out] recv Receive buffer
         @param[in] send Send buffer
         @param[in] dir Direction in which we are sending (forwards OR backwards only)
      */
      void exchange(void **recv, void **send, QudaDirection dir) const;

      /**
         Compute the required extended ghost zone sizes and offsets
         @param[in] R Radius of the ghost zone
         @param[in] no_comms_fill If true we create a full halo
         regardless of partitioning
         @param[in] bidir Is this a bi-directional exchange - if not
         then we alias the fowards and backwards offsetss
      */
      void createGhostZone(const int *R, bool no_comms_fill, bool bidir = true) const;

      /**
         @brief Set the vol_string and aux_string for use in tuning
      */
      void setTuningString();

  public:
    MiscField(const MiscFieldParam &param);
    virtual ~MiscField();

    virtual void exchangeGhost(QudaLinkDirection = QUDA_LINK_BACKWARDS) = 0;
    virtual void injectGhost(QudaLinkDirection = QUDA_LINK_BACKWARDS) = 0;

    size_t Length() const { return length; }
    int Ncol() const { return nCol; }
    int Nrow() const { return nRow; }
    QudaMiscFieldOrder Order() const { return order; }
    QudaTboundary TBoundary() const { return t_boundary; }
    QudaMiscFieldOrder FieldOrder() const { return order; }
    QudaFieldGeometry Geometry() const { return geometry; }

    /**
     * Define the parameter type for this field.
     */
    using param_type = MiscFieldParam;

    int Nface() const { return nFace; }

    /**
       @brief This routine will populate the border / halo region of a
       misc field that has been created using copyExtendedMisc.
       @param R The thickness of the extended region in each dimension
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    virtual void exchangeExtendedGhost(const int *R, bool no_comms_fill = false) = 0;

    /**
       @brief This routine will populate the border / halo region
       of a misc field that has been created using copyExtendedMisc.
       Overloaded variant that will start and stop a comms profile.
       @param R The thickness of the extended region in each dimension
       @param profile TimeProfile intance which will record the time taken
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    virtual void exchangeExtendedGhost(const int *R, TimeProfile &profile, bool no_comms_fill = false) = 0;

    void checkField(const LatticeField &) const;

    /**
       This function returns true if the field is stored in an
       internal field order for the given precision.
    */
    bool isNative() const { return misc::isNative(order, precision); }

    size_t Bytes() const { return bytes; }

    size_t TotalBytes() const { return bytes; }

    virtual void* Misc_p() { errorQuda("Not implemented"); return (void*)0;}
    virtual void* Even_p() { errorQuda("Not implemented"); return (void*)0;}
    virtual void* Odd_p() { errorQuda("Not implemented"); return (void*)0;}

    virtual const void* Misc_p() const { errorQuda("Not implemented"); return (void*)0;}
    virtual const void* Even_p() const { errorQuda("Not implemented"); return (void*)0;}
    virtual const void* Odd_p() const { errorQuda("Not implemented"); return (void*)0;}

    virtual int full_dim(int d) const { return x[d]; }

    const void** Ghost() const {
      if ( isNative() ) errorQuda("No ghost zone pointer for quda-native misc fields");
      return (const void**)ghost;
    }

    void** Ghost() {
      if ( isNative() ) errorQuda("No ghost zone pointer for quda-native misc fields");
      return ghost;
    }

    /**
       Set all field elements to zero (virtual)
    */
    virtual void zero() = 0;

    /**
     * Generic misc field copy
     * @param[in] src Source from which we are copying
     */
    virtual void copy(const MiscField &src) = 0;

    /**
       @brief Compute the L1 norm of the field
       @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
       @return L1 norm
     */
    double norm1(int dim = -1, bool fixed = false) const;

    /**
       @brief Compute the L2 norm squared of the field
       @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
       @return L2 norm squared
     */
    double norm2(int dim = -1, bool fixed = false) const;

    /**
       @brief Compute the absolute maximum of the field (Linfinity norm)
       @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
       @return Absolute maximum value
     */
    double abs_max(int dim = -1, bool fixed = false) const;

    /**
       @brief Compute the absolute minimum of the field
       @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
       @return Absolute minimum value
     */
    double abs_min(int dim = -1, bool fixed = false) const;

    /**
       Compute checksum of this misc field: this uses a XOR-based checksum method
       @param[in] mini Whether to compute a mini checksum or global checksum.
       A mini checksum only computes the checksum over a subset of the lattice
       sites and is to be used for online comparisons, e.g., checking
       a field has changed with a global update algorithm.
       @return checksum value
     */
    uint64_t checksum(bool mini = false) const;

    /**
       @brief Create the misc field, with meta data specified in the
       parameter struct.
       @param param Parameter struct specifying the misc field
       @return Pointer to allcoated misc field
    */
    static MiscField* Create(const MiscFieldParam &param);

  };

  class cudaMiscField : public MiscField {

  private:
    void *misc;
    void *misc_h; // mapped-memory pointer when allocating on the host
    void *even;
    void *odd;

    /**
       @brief Initialize the padded region to 0
     */
    void zeroPad();

  public:
    cudaMiscField(const MiscFieldParam &);
    virtual ~cudaMiscField();

    /**
       @brief Exchange the ghost and store store in the padded region
     */
    void exchangeGhost();

    /**
       @brief The opposite of exchangeGhost: take the ghost zone on x,
       send to node x-1, and inject back into the field
     */
    void injectGhost();

    /**
       @brief Create the communication handlers and buffers
       @param[in] R The thickness of the extended region in each dimension
       @param[in] no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
       @param[in] bidir Whether to allocate communication buffers to
       allow for simultaneous bi-directional exchange.  If false, then
       the forwards and backwards buffers will alias (saving memory).
       FIXME: the bidir is probably never necessary for this type of field
    */
    void createComms(const int *R, bool no_comms_fill, bool bidir=true);

    /**
       @brief Allocate the ghost buffers
       @param[in] R The thickness of the extended region in each dimension
       @param[in] no_comms_fill Do local exchange to fill out the extended
       @param[in] bidir Is this a bi-directional exchange - if not
       then we alias the fowards and backwards offsetss
       region in non-partitioned dimensions
       FIXME: the bidir is probably never necessary for this type of field
    */
    void allocateGhostBuffer(const int *R, bool no_comms_fill, bool bidir=true) const;

    /**
       @brief Start the receive communicators
       @param[in] dim The communication dimension
       @param[in] dir The communication direction (0=backwards, 1=forwards)
    */
    void recvStart(int dim, int dir);

    /**
       @brief Start the sending communicators
       @param[in] dim The communication dimension
       @param[in] dir The communication direction (0=backwards, 1=forwards)
       @param[in] stream_p Pointer to CUDA stream to post the
       communication in (if 0, then use null stream)
    */
    void sendStart(int dim, int dir, const qudaStream_t &stream_p);

    /**
       @brief Wait for communication to complete
       @param[in] dim The communication dimension
       @param[in] dir The communication direction (0=backwards, 1=forwards)
    */
    void commsComplete(int dim, int dir);

    /**
       @brief This does routine will populate the border / halo region of a
       misc field that has been created using copyExtendedMisc.
       @param R The thickness of the extended region in each dimension
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    void exchangeExtendedGhost(const int *R, bool no_comms_fill=false);

    /**
       @brief This does routine will populate the border / halo region
       of a misc field that has been created using copyExtendedMisc.
       Overloaded variant that will start and stop a comms profile.
       @param R The thickness of the extended region in each dimension
       @param profile TimeProfile intance which will record the time taken
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    void exchangeExtendedGhost(const int *R, TimeProfile &profile, bool no_comms_fill=false);

    /**
     * Generic misc field copy
     * @param[in] src Source from which we are copying
     */
    void copy(const MiscField &src);

    /**
       @brief Download into this field from a CPU field
       @param[in] cpu The CPU field source
    */
    void loadCPUField(const cpuMiscField &cpu);

    /**
       @brief Download into this field from a CPU field.  Overloaded
       variant that includes profiling
       @param[in] cpu The CPU field source
       @param[in] profile Time profile to record the transfer
    */
    void loadCPUField(const cpuMiscField &cpu, TimeProfile &profile);

    /**
       @brief Upload from this field into a CPU field
       @param[out] cpu The CPU field source
    */
    void saveCPUField(cpuMiscField &cpu) const;

    /**
       @brief Upload from this field into a CPU field.  Overloaded
       variant that includes profiling.
       @param[out] cpu The CPU field source
       @param[in] profile Time profile to record the transfer
    */
    void saveCPUField(cpuMiscField &cpu, TimeProfile &profile) const;

    // (ab)use with care
    void* Misc_p() { return misc; }
    void* Even_p() { return even; }
    void* Odd_p() { return odd; }

    const void* Misc_p() const { return misc; }
    const void* Even_p() const { return even; }
    const void *Odd_p() const { return odd; }

    /**
      @brief Copy all contents of the field to a host buffer.
      @param[in] the host buffer to copy to.
    */
    virtual void copy_to_buffer(void *buffer) const;

    /**
      @brief Copy all contents of the field from a host buffer to this field.
      @param[in] the host buffer to copy from.
    */
    virtual void copy_from_buffer(void *buffer);

    void setMisc(void* _misc); //only allowed when create== QUDA_REFERENCE_FIELD_CREATE

    /**
       Set all field elements to zero
    */
    void zero();

    /**
       @brief Backs up the cudaMiscField to CPU memory
    */
    void backup() const;

    /**
       @brief Restores the cudaMiscField to CUDA memory
    */
    void restore() const;

    /**
      @brief If managed memory and prefetch is enabled, prefetch
      the misc field and buffers to the CPU or the GPU
      @param[in] mem_space Memory space we are prefetching to
      @param[in] stream Which stream to run the prefetch in (default 0)
    */
    void prefetch(QudaFieldLocation mem_space, qudaStream_t stream = device::get_default_stream()) const;
  };

  class cpuMiscField : public MiscField {

    friend void cudaMiscField::copy(const MiscField &cpu);
    friend void cudaMiscField::loadCPUField(const cpuMiscField &cpu);
    friend void cudaMiscField::saveCPUField(cpuMiscField &cpu) const;

  private:
    void **misc; // the actual misc field

  public:
    /**
       @brief Constructor for cpuMiscField from a MiscFieldParam
       @param[in,out] param Parameter struct - note that in the case
       that we are wrapping host-side extended fields, this param is
       modified for subsequent creation of fields that are not
       extended.
    */
    cpuMiscField(const MiscFieldParam &param);
    virtual ~cpuMiscField();

    /**
       @brief Exchange the ghost and store store in the padded region
       @param[in] link_direction Which links are we extracting: this
       flag only applies to bi-directional coarse-link fields
     */
    void exchangeGhost(QudaLinkDirection link_direction = QUDA_LINK_BACKWARDS);

    /**
       @brief The opposite of exchangeGhost: take the ghost zone on x,
       send to node x-1, and inject back into the field
       @param[in] link_direction Which links are we injecting: this
       flag only applies to bi-directional coarse-link fields
     */
    void injectGhost(QudaLinkDirection link_direction = QUDA_LINK_BACKWARDS);

    /**
       @brief This does routine will populate the border / halo region of a
       misc field that has been created using copyExtendedMisc.

       @param R The thickness of the extended region in each dimension
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimenions
    */
    void exchangeExtendedGhost(const int *R, bool no_comms_fill=false);

    /**
       @brief This does routine will populate the border / halo region
       of a misc field that has been created using copyExtendedMisc.
       Overloaded variant that will start and stop a comms profile.
       @param R The thickness of the extended region in each dimension
       @param profile TimeProfile intance which will record the time taken
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    void exchangeExtendedGhost(const int *R, TimeProfile &profile, bool no_comms_fill=false);

    /**
     * Generic misc field copy
     * @param[in] src Source from which we are copying
     */
    void copy(const MiscField &src);

    void* Misc_p() { return misc; }
    const void* Misc_p() const { return misc; }

    /**
      @brief Copy all contents of the field to a host buffer.
      @param[in] the host buffer to copy to.
    */
    virtual void copy_to_buffer(void *buffer) const;

    /**
      @brief Copy all contents of the field from a host buffer to this field.
      @param[in] the host buffer to copy from.
    */
    virtual void copy_from_buffer(void *buffer);

    void setMisc(void** _misc); //only allowed when create== QUDA_REFERENCE_FIELD_CREATE

    /**
       Set all field elements to zero
    */
    void zero();

    /**
       @brief Backs up the cpuMiscField
    */
    void backup() const;

    /**
       @brief Restores the cpuMiscField
    */
    void restore() const;
  };

  /**
     @brief This is a debugging function, where we cast a misc field
     into a spinor field so we can compute its L1 norm.
     @param m The misc field that we want the norm of
     @return The L1 norm of the misc field
  */
  double norm1(const MiscField &m);

  /**
     @brief This is a debugging function, where we cast a misc field
     into a spinor field so we can compute its L2 norm.
     @param m The misc field that we want the norm of
     @return The L2 norm squared of the misc field
  */
  double norm2(const MiscField &m);

  /**
     @brief Scale the misc field by the scalar a.
     @param[in] a scalar multiplier
     @param[in] m The misc field we want to multiply
   */
  void ax(const double &a, MiscField &m);

  /**
     This function is used for  extracting the misc ghost zone from a
     misc field array.  Defined in copy_misc.cu.
     @param out The output field to which we are copying
     @param in The input field from which we are copying
     @param location The location of where we are doing the copying (CPU or CUDA)
     @param Out The output buffer (optional)
     @param In The input buffer (optional)
     @param ghostOut The output ghost buffer (optional)
     @param ghostIn The input ghost buffer (optional)
     @param type The type of copy we doing (0 body and ghost else ghost only)
  */
  void copyGenericMisc(MiscField &out, const MiscField &in, QudaFieldLocation location, void *Out = 0, void *In = 0,
                        void **ghostOut = 0, void **ghostIn = 0, int type = 0);

  /**
    @brief This function is used for copying from a source misc field to a destination misc field
      with an offset.
    @param out The output field to which we are copying
    @param in The input field from which we are copying
    @param offset The offset for the larger field between out and in.
    @param pc_type Whether the field order uses 4d or 5d even-odd preconditioning.
 */
  void copyFieldOffset(MiscField &out, const MiscField &in, CommKey offset, QudaPCType pc_type);

  /**
     This function is used for copying the misc field into an
     extended misc field.  Defined in copy_extended_misc.cu.
     @param out The extended output field to which we are copying
     @param in The input field from which we are copying
     @param location The location of where we are doing the copying (CPU or CUDA)
     @param Out The output buffer (optional)
     @param In The input buffer (optional)
  */
  void copyExtendedMisc(MiscField &out, const MiscField &in,
			 QudaFieldLocation location, void *Out=0, void *In=0);

  /**
     This function is used for creating an exteneded misc field from the input,
     and copying the misc field into the extended misc field.  Defined in lib/misc_field.cpp.
     @param in The input field from which we are extending
     @param R By how many do we want to extend the misc field in each direction
     @param profile The `TimeProfile`
     @param redundant_comms
     @param recon The reconsturction type
     @return the pointer to the extended misc field
  */
  cudaMiscField *createExtendedMisc(cudaMiscField &in, const int *R, TimeProfile &profile,
                                      bool redundant_comms = false, QudaReconstructType recon = QUDA_RECONSTRUCT_INVALID);

  /**
     This function is used for creating an exteneded (cpu) misc field from the input,
     and copying the misc field into the extended misc field.  Defined in lib/misc_field.cpp.
     @param in The input field from which we are extending
     @param R By how many do we want to extend the misc field in each direction
     @return the pointer to the extended misc field
  */
  cpuMiscField *createExtendedMisc(void **misc, QudaMiscParam &misc_param, const int *R);

  /**
     This function is used for  extracting the misc ghost zone from a
     misc field array.  Defined in extract_misc_ghost.cu.
     @param m The misc field from which we want to extract the ghost zone
     @param ghost The array where we want to pack the ghost zone into
     @param extract Where we are extracting into ghost or injecting from ghost
     @param offset By default we exchange the nDim site-vector of
     links in the first nDim dimensions; offset allows us to instead
     exchange the links in nDim+offset dimensions.  This is used to
     faciliate sending bi-directional links which is needed for the
     coarse links.
  */
  void extractMiscGhost(const MiscField &m, void **ghost, bool extract=true, int offset=0);

  /**
     This function is used for extracting the extended misc ghost
     zone from a misc field array.  Defined in
     extract_misc_ghost_extended.cu.
     @param m The misc field from which we want to extract/pack the ghost zone
     @param dim The dimension in which we are packing/unpacking
     @param R array holding the radius of the extended region
     @param ghost The array where we want to pack/unpack the ghost zone into/from
     @param extract Whether we are extracting into ghost or injecting from ghost
  */
  void extractExtendedMiscGhost(const MiscField &m, int dim, const int *R, void **ghost, bool extract);

  /**
     Apply the staggered phase factor to the misc field.
     @param[in] m The misc field to which we apply the staggered phase factors
  */
  void applyMiscPhase(MiscField &m);

  /**
     Compute XOR-based checksum of this misc field: each misc field entry is
     converted to type uint64_t, and compute the cummulative XOR of these values.
     @param[in] mini Whether to compute a mini checksum or global checksum.
     A mini checksum only computes over a subset of the lattice
     sites and is to be used for online comparisons, e.g., checking
     a field has changed with a global update algorithm.
     @return checksum value
  */
  uint64_t Checksum(const MiscField &m, bool mini=false);

} // namespace quda
