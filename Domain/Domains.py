


import Domain.Domain as dm
import Domain.Coordinates as cd

dms = []
for c in cd.domain_Coors:
    d = dm.Domain(c[0],c[1],c[2],c[3],c[4],c[5])
    dms.append(d)