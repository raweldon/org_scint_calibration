import numpy as np

'''
mcnpx-polimi collision record data types: field name and type
'''
# ASCII
dtype = [
            (   'history'       ,   np.int   ),  # history number
            (   'particle'      ,   np.int   ),  # particle number
            (   'projectile'    ,   np.int   ),  # projectile type (1 = neutron, 2 = photon)
            (   'interaction'   ,   np.int   ),  # interaction type (-99 = neutron elastic scatter, -1 = neutron inelastic scatter, 0 = neutron or photon absoprtion, 1 = photon Compton scatter)
            (   'target'        ,   np.int   ),  # target ZAID
            (   'cell'          ,   np.int   ),  # cell number
            (   'deltaEnergy'   ,   np.float ),  # energy deposition (MeV)
            (   'time'          ,   np.float ),  # time (shakes)
            (   'x'             ,   np.float ),  # x-position (cm)
            (   'y'             ,   np.float ),  # y-position (cm)
            (   'z'             ,   np.float ),  # z-position (cm)
            (   'weight'        ,   np.float ),  # particle weight
            (   'generation'    ,   np.int   ),  # fission chain-reaction generation
            (   'scatters'      ,   np.int   ),  # number of scatters
            (   'code'          ,   np.int   ),  # 'special' code
            (   'energy'        ,   np.float )   # incident energy (MeV)
        ]

# Binary
#some compilers have history as int32 and others as int64 (long)
#dtype68 = [
#            (   'leader'        ,   np.int32   ),  # write leader (a fortran thing)
#            (   'history'       ,   np.int32   ),  # history number
#            (   'particle'      ,   np.int32   ),  # particle number
#            (   'projectile'    ,   np.int32   ),  # projectile type (1 = neutron, 2 = photon)
#            (   'interaction'   ,   np.int32   ),  # interaction type (-99 = neutron elastic scatter, -1 = neutron inelastic scatter, 0 = neutron or photon absoprtion, 1 = photon Compton scatter)
#            (   'target'        ,   np.int32   ),  # target ZAID
#            (   'cell'          ,   np.int32   ),  # cell number
#            (   'deltaEnergy'   ,   np.float32 ),  # energy deposition (MeV)
#            (   'time'          ,   np.float64 ),  # time (shakes)
#            (   'x'             ,   np.float32 ),  # x-position (cm)
#            (   'y'             ,   np.float32 ),  # y-position (cm)
#            (   'z'             ,   np.float32 ),  # z-position (cm)
#            (   'weight'        ,   np.float32 ),  # particle weight
#            (   'generation'    ,   np.int32   ),  # fission chain-reaction generation
#            (   'scatters'      ,   np.int32   ),  # number of scatters
#            (   'code'          ,   np.int32   ),  # 'special' code
#            (   'energy'        ,   np.float32 ),  # incident energy (MeV)
#            (   'trailer'       ,   np.int32   )   # write trailer (a fortran thing)
#        ]

dtype72 = [
            (   'leader'        ,   np.int32   ),  # write leader (a fortran thing)
            (   'history'       ,   np.int64   ),  # history number
            (   'particle'      ,   np.int32   ),  # particle number
            (   'projectile'    ,   np.int32   ),  # projectile type (1 = neutron, 2 = photon)
            (   'interaction'   ,   np.int32   ),  # interaction type (-99 = neutron elastic scatter, -1 = neutron inelastic scatter, 0 = neutron or photon absoprtion, 1 = photon Compton scatter)
            (   'target'        ,   np.int32   ),  # target ZAID
            (   'cell'          ,   np.int32   ),  # cell number
            (   'deltaEnergy'   ,   np.float32 ),  # energy deposition (MeV)
            (   'time'          ,   np.float64 ),  # time (shakes)
            (   'x'             ,   np.float32 ),  # x-position (cm)
            (   'y'             ,   np.float32 ),  # y-position (cm)
            (   'z'             ,   np.float32 ),  # z-position (cm)
            (   'weight'        ,   np.float32 ),  # particle weight
            (   'generation'    ,   np.int32   ),  # fission chain-reaction generation
            (   'scatters'      ,   np.int32   ),  # number of scatters
            (   'code'          ,   np.int32   ),  # 'special' code
            (   'energy'        ,   np.float32 ),  # incident energy (MeV)
            (   'trailer'       ,   np.int32   )   # write trailer (a fortran thing)
        ]

'''
load collisions from text format mcnpx-polimi collision log file
returns record array
'''
#ASCII
def loadtxt(filename):
    collision = np.loadtxt(filename, dtype = dtype)
    return collision.view(np.recarray)

#Binary
def fromfile(filename):
    #del dtype1[0]
    #del dtype1[-1]
    collision = np.fromfile(filename, dtype = dtype72)
    return collision.view(np.recarray)
    
