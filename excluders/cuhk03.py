import os
import re

import numpy as np


class Excluder:
    """
    In the Market1501 evaluation, we need to exclude both the same PID in
    the same camera (CID), as well as "junk" images (PID=-1).
    """
    def __init__(self, gallery_fids):
        # Setup a regexp for extracing the PID and camera (CID) form a FID.
        self.regexp = re.compile('(\S+)_c(\d+)s(\d+)_.*')

        # Parse the gallery_set
        self.gallery_cpids,self.gallery_incpids,self.gallery_cids = self._parse(gallery_fids)

    def __call__(self, query_fids):
        # Extract both the PIDs and CIDs from the query filenames:
        query_cpids, query_incpids, query_cids = self._parse(query_fids)

        # Ignore same pid image within the same camera
        cpid_matches = self.gallery_cpids[None] == query_cpids[:,None]
        incpid_matches = self.gallery_incpids[None] == query_incpids[:,None]
        cid_matches = self.gallery_cids[None] == query_cids[:,None]
        mask0 = np.logical_and(cpid_matches, cid_matches)
        mask = np.logical_and(mask0, incpid_matches)

        # Remove all "junk" with the -1 pid.
        #junk_images = np.repeat(self.gallery_pids[None] == '-1', len(query_pids), 0)
        #mask = np.logical_or(mask, junk_images)

        return mask

    def _parse(self, fids):
        """ Return the PIDs and CIDs extracted from the FIDs. """
        cpids = []
        incpids = []
        cids = []
        for fid in fids:
            filename = fid.split('/')[-1]
            filename = filename.split('_')
            cpid = filename[0]
            incpid = filename[1]
            cid= filename[2]
            cpids.append(cpid)
            incpids.append(incpid)
            cids.append(cid)
        return np.asarray(cpids),np.asarray(incpids), np.asarray(cids)
