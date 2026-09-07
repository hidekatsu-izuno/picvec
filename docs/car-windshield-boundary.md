# Car windshield boundary regression

The notch in the far red windshield rim was introduced by
`refine_thin_paint_ownership`, before curve fitting. The eight source sites
at (354–355, 440), (353–354, 441), (352–353, 442), and (351–352, 443)
belonged to a red fragment with canonical RGB approximately (189, 19, 15).
The thin-Paint refinement reassigned all eight sites to the background.

Repeated boundary fragments are grouped to identify their durable incident
faces. However, the old assignment then chose an owner using only each
fragment's direct durable contacts. The neighbouring red fragments had no
interior, leaving the background as this fragment's sole eligible direct
contact. No comparison with the better colour match elsewhere in the
family prevented the transfer. Curve fitting subsequently reproduced the
damaged silhouette; removing structural ink did not remove the notch.

The assignment now checks the proposed owner's colour against every
durable parent established by the family. If a better matching parent is
screened by another small fragment, the original fragment is retained
instead of being forced into an incompatible directly touching face.

`boundary_phase_cannot_discard_colour_when_its_matching_parent_is_screened`
reproduces the topology using a synthetic red boundary. It fails before
the fix, preserves the screened red fragment afterward, and also checks
that unscreened fragments can still join their matching parent.

In the regenerated car, all eight sites retain their red ownership and the
rendered rim is continuous. The headlight highlight regression also remains
fixed. The library test suite passes with 234 passed and 4 ignored tests.
