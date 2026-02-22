"""
Script to generate synthetic legal PDF documents for testing and demo purposes.
Run with: python sample_docs/generate_samples.py
Requires: pip install reportlab
"""

import os
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("reportlab not installed -- generating .txt files instead")


SAMPLE_DOCUMENTS = [
    {
        "filename": "contract_acme_services_2023.pdf",
        "title": "PROFESSIONAL SERVICES AGREEMENT",
        "content": """PROFESSIONAL SERVICES AGREEMENT

This Professional Services Agreement ("Agreement") is entered into as of March 15, 2023 ("Effective Date"),
by and between Acme Corporation, a Delaware corporation ("Client"), having its principal place of business at
100 Commerce Street, Wilmington, DE 19801, and LexTech Solutions LLC, a New York limited liability company
("Service Provider"), having its principal place of business at 200 Park Avenue, New York, NY 10166.

1. SERVICES
Service Provider agrees to provide Client with legal technology consulting services as described in
Exhibit A attached hereto ("Services"). Service Provider shall perform the Services in a professional
and workmanlike manner consistent with industry standards.

2. TERM
This Agreement shall commence on the Effective Date and continue for a period of twelve (12) months
unless earlier terminated in accordance with Section 9 of this Agreement.

3. COMPENSATION
Client shall pay Service Provider a monthly retainer fee of USD $15,000 (fifteen thousand dollars),
due and payable within thirty (30) days of invoice. Late payments shall accrue interest at the rate
of 1.5% per month.

4. CONFIDENTIALITY
Each party agrees to keep confidential all non-public information disclosed by the other party
("Confidential Information") and to use such Confidential Information solely for the purposes of
this Agreement. This obligation shall survive termination for a period of five (5) years.

5. INDEMNIFICATION
Service Provider shall indemnify, defend, and hold harmless Client and its officers, directors,
employees, and agents from and against any and all claims, damages, losses, costs, and expenses
(including reasonable attorneys' fees) arising out of or related to Service Provider's negligence,
willful misconduct, or breach of this Agreement.

6. LIMITATION OF LIABILITY
In no event shall either party be liable to the other for any indirect, incidental, special,
exemplary, or consequential damages, even if advised of the possibility of such damages.
The total cumulative liability of Service Provider under this Agreement shall not exceed the
total fees paid by Client in the twelve (12) months preceding the claim.

7. INTELLECTUAL PROPERTY
All work product created by Service Provider specifically for Client under this Agreement shall
be considered "work made for hire" and shall be owned exclusively by Client upon full payment.

8. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the State of
New York, without regard to conflicts of law principles. Any disputes shall be resolved by
binding arbitration in New York City under the rules of JAMS.

9. TERMINATION
Either party may terminate this Agreement upon thirty (30) days written notice. Client may
terminate immediately for cause if Service Provider materially breaches this Agreement and
fails to cure such breach within fifteen (15) days of written notice.

10. FORCE MAJEURE
Neither party shall be liable for any delay or failure to perform its obligations under this
Agreement due to causes beyond its reasonable control, including acts of God, war, terrorism,
pandemics, government orders, or natural disasters ("Force Majeure Event").

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

ACME CORPORATION                    LEXTECH SOLUTIONS LLC
By: ______________________          By: ______________________
Name: John Smith                    Name: Sarah Johnson
Title: Chief Executive Officer      Title: Managing Member
Date: March 15, 2023                Date: March 15, 2023

EXHIBIT A – SCOPE OF SERVICES
1. Legal technology assessment and roadmap development
2. Contract lifecycle management system implementation
3. AI-powered document review configuration and training
4. Staff training and change management support
Client Matter ID: ACM-2023-0315
""",
    },
    {
        "filename": "case_file_henderson_v_blackwood_2022.pdf",
        "title": "CASE FILE: Henderson v. Blackwood Industries",
        "content": """CONFIDENTIAL CASE FILE

Case Name: Henderson v. Blackwood Industries, Inc.
Case Number: 2022-CV-04521
Court: United States District Court, Southern District of New York
Filed: June 8, 2022
Client: Robert Henderson
Matter ID: HEN-2022-0608

CASE SUMMARY
Plaintiff Robert Henderson ("Henderson") brings this action against Blackwood Industries, Inc.
("Blackwood") for wrongful termination, breach of employment contract, and violation of the
Sarbanes-Oxley Act whistleblower protection provisions, 18 U.S.C. § 1514A.

FACTS
Henderson was employed by Blackwood as Vice President of Finance from January 2018 until his
termination on April 30, 2022. During Q4 2021, Henderson discovered what he reasonably believed
to be securities fraud: Blackwood's CFO, Marcus Reeves, had directed the accounting team to
improperly capitalize operating expenses, inflating reported earnings by approximately $4.2 million.

Henderson reported his concerns internally to the Audit Committee on February 14, 2022. On
March 1, 2022, he filed a complaint with the Securities and Exchange Commission. Sixty days later,
on April 30, 2022, Blackwood terminated Henderson's employment, citing "restructuring."

LEGAL CLAIMS
Count I: Wrongful Termination in Violation of Sarbanes-Oxley Act (18 U.S.C. § 1514A)
  - Henderson engaged in protected activity by reporting suspected securities violations
  - The 60-day proximity between the SEC complaint and termination creates a strong inference
    of retaliatory intent
  - Blackwood's stated reason (restructuring) is pretextual -- Henderson's position was
    subsequently filled by an internal candidate

Count II: Breach of Employment Contract
  - Henderson's employment agreement dated January 5, 2018 contains a clause requiring
    "cause" for termination, defined in Section 4.2 as gross misconduct, conviction of a felony,
    or willful neglect of duties
  - None of these conditions were met
  - Henderson is entitled to 12 months severance pay per Section 7.1 of the agreement

Count III: Defamation
  - Blackwood's general counsel stated to Henderson's subsequent prospective employer that
    Henderson was "terminated for financial irregularities" -- a false statement made with
    knowledge of its falsity or reckless disregard for the truth

DAMAGES
  - Back pay: approximately $340,000 (10 months at $408,000 annual salary)
  - Front pay: approximately $816,000 (estimated 24 months to equivalent re-employment)
  - Compensatory damages for emotional distress: TBD
  - Punitive damages under SOX: up to $250,000
  - Attorneys' fees and costs

KEY DOCUMENTS
  1. Employment Agreement (Jan 5, 2018) -- signed by Henderson and CEO Patricia Blackwood
  2. Internal Audit Committee memo (Feb 14, 2022)
  3. SEC whistleblower complaint (Mar 1, 2022) -- confidential
  4. Termination letter (Apr 30, 2022)
  5. Blackwood Q4 2021 financial statements (allegedly fraudulent)
  6. Email chain between CFO Reeves and Accounting team (subpoenaed)

NEXT STEPS
  - Discovery cutoff: December 1, 2022
  - Depose CFO Marcus Reeves and CEO Patricia Blackwood
  - Retain forensic accounting expert to quantify earnings manipulation
  - File motion for summary judgment by January 15, 2023
""",
    },
    {
        "filename": "nda_globex_stellartech_2023.pdf",
        "title": "MUTUAL NON-DISCLOSURE AGREEMENT",
        "content": """MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of September 1, 2023,
between Globex International Ltd., a United Kingdom corporation ("Party A"), and StellarTech
Ventures, Inc., a California corporation ("Party B"). Each of Party A and Party B is referred
to individually as a "Party" and collectively as the "Parties."

1. PURPOSE
The Parties wish to explore a potential business partnership involving the joint development
of artificial intelligence tools for legal document analysis (the "Proposed Transaction").
In connection with evaluating the Proposed Transaction, each Party may disclose or make
available to the other certain non-public, proprietary, or confidential information.

2. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means any non-public information that a Party designates as
confidential or that a reasonable person would understand to be confidential given the nature
of the information and circumstances of disclosure, including but not limited to:
  (a) trade secrets, inventions, patent applications, and technical data;
  (b) business plans, financial projections, and customer lists;
  (c) software, algorithms, and model weights (including large language model parameters);
  (d) the existence and terms of the Proposed Transaction.

Confidential Information does NOT include information that:
  (i) is or becomes publicly available without breach of this Agreement;
  (ii) was rightfully known by the Receiving Party prior to disclosure;
  (iii) is rightfully obtained from a third party without restriction; or
  (iv) is independently developed by the Receiving Party without use of Confidential Information.

3. OBLIGATIONS
Each Party (the "Receiving Party") agrees to:
  (a) hold the other Party's Confidential Information in strict confidence;
  (b) use Confidential Information solely to evaluate or pursue the Proposed Transaction;
  (c) limit disclosure to employees and advisors with a need to know who are bound by
      obligations at least as protective as those in this Agreement;
  (d) protect Confidential Information using at least the same degree of care it uses to
      protect its own confidential information, but no less than reasonable care.

4. TERM
This Agreement shall remain in effect for three (3) years from the Effective Date.
The confidentiality obligations with respect to trade secrets shall survive indefinitely.

5. RETURN OR DESTRUCTION
Upon written request or termination of discussions, each Party shall promptly return or
certify destruction of all Confidential Information and copies thereof, except as required
by law or pursuant to automated backup systems.

6. INJUNCTIVE RELIEF
Each Party acknowledges that a breach of this Agreement may cause irreparable harm for
which monetary damages would be an inadequate remedy. Accordingly, either Party shall be
entitled to seek equitable relief, including injunction and specific performance, without
the requirement of posting bond.

7. GOVERNING LAW AND JURISDICTION
This Agreement shall be governed by the laws of the State of Delaware. The Parties consent
to the exclusive jurisdiction of the state and federal courts located in Wilmington, Delaware.

8. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the Parties with respect to its
subject matter and supersedes all prior negotiations, representations, or agreements.

Party A: GLOBEX INTERNATIONAL LTD.     Party B: STELLARTECH VENTURES, INC.
By: _________________________           By: _________________________
Name: Elena Marchetti                   Name: David Kim
Title: Chief Legal Officer              Title: Chief Executive Officer
Date: September 1, 2023                 Date: September 1, 2023
""",
    },
    {
        "filename": "brief_motion_summary_judgment_2022.pdf",
        "title": "MEMORANDUM OF LAW IN SUPPORT OF MOTION FOR SUMMARY JUDGMENT",
        "content": """UNITED STATES DISTRICT COURT
SOUTHERN DISTRICT OF NEW YORK

PRISM ANALYTICS, INC.,
    Plaintiff,
                                        Case No. 22-CV-09874
    v.
                                        MEMORANDUM OF LAW IN SUPPORT OF
VECTOR DATA SYSTEMS, LLC,               DEFENDANT'S MOTION FOR SUMMARY
    Defendant.                          JUDGMENT

PRELIMINARY STATEMENT

Defendant Vector Data Systems, LLC ("Vector Data") respectfully submits this memorandum in
support of its motion for summary judgment pursuant to Federal Rule of Civil Procedure 56.
Summary judgment is warranted because Plaintiff Prism Analytics, Inc. ("Prism") cannot
establish a genuine dispute of material fact on any of its claims.

STATEMENT OF FACTS

Prism alleges that Vector Data misappropriated Prism's proprietary data analytics methodology
and incorporated it into Vector Data's competing "DataPulse" platform. The undisputed record
shows otherwise:

1. Vector Data independently developed DataPulse beginning in Q1 2020 -- before any contact
   with Prism -- as evidenced by git commit logs, engineering design documents, and testimony
   from Vector Data's engineering team.

2. Prism's asserted trade secret -- its "weighted confidence interval methodology" -- was
   disclosed in Prism's 2019 white paper published openly on its website and in a peer-reviewed
   academic journal, destroying any trade secret protection.

3. The functionality Prism identifies as allegedly misappropriated (adaptive threshold
   calibration) is covered by Vector Data's U.S. Patent No. 11,234,567, filed in March 2020.

LEGAL STANDARD

Summary judgment is appropriate when "there is no genuine dispute as to any material fact
and the movant is entitled to judgment as a matter of law." Fed. R. Civ. P. 56(a); see
Celotex Corp. v. Catrett, 477 U.S. 317, 322 (1986). The non-moving party must present
"significant probative evidence" to defeat a properly supported motion. Anderson v. Liberty
Lobby, Inc., 477 U.S. 242, 249 (1986).

ARGUMENT

I. PRISM'S TRADE SECRET CLAIMS FAIL BECAUSE THE ALLEGED SECRET WAS PUBLICLY DISCLOSED

To prevail on a trade secret misappropriation claim under the Defend Trade Secrets Act
(18 U.S.C. § 1836), a plaintiff must demonstrate that the information constituted a trade
secret at the time of misappropriation. A trade secret must be kept reasonably secret by
its holder. Prism's publication of its methodology in 2019 -- with full technical detail --
destroyed any trade secret protection before Prism and Vector Data ever discussed a partnership.

II. VECTOR DATA INDEPENDENTLY DEVELOPED DATAPULSE

Even if Prism's methodology had retained trade secret status, independent development is a
complete defense under 18 U.S.C. § 1839(6)(B). The undisputed engineering records establish
that Vector Data's team independently arrived at similar techniques through their own research.

III. PATENT PREEMPTION

Prism's state-law unfair competition claim is preempted to the extent it is based on the
same conduct as the patent misappropriation claim. See Bonito Boats, Inc. v. Thunder Craft
Boats, Inc., 489 U.S. 141, 156 (1989).

CONCLUSION

For the foregoing reasons, Vector Data respectfully requests that this Court grant summary
judgment in its favor on all counts, and award Vector Data its reasonable attorneys' fees
and costs as the prevailing party under 18 U.S.C. § 1836(b)(3)(D).

Respectfully submitted,
MORRISON & HARTWELL LLP

By: /s/ Katherine A. Morrison
Katherine A. Morrison
Counsel for Defendant Vector Data Systems, LLC
Date: November 10, 2022
""",
    },
    {
        "filename": "employment_agreement_cto_2023.pdf",
        "title": "EXECUTIVE EMPLOYMENT AGREEMENT – CHIEF TECHNOLOGY OFFICER",
        "content": """EXECUTIVE EMPLOYMENT AGREEMENT

This Executive Employment Agreement ("Agreement") is made and entered into as of January 1, 2023
("Effective Date"), by and between NovaStar Financial Group, Inc., a Delaware corporation
("Company"), and Dr. Amara Osei ("Executive").

RECITALS
The Company desires to employ Executive as its Chief Technology Officer, and Executive desires
to accept such employment, on the terms and conditions set forth herein.

1. POSITION AND DUTIES
Executive shall serve as Chief Technology Officer ("CTO") reporting directly to the Chief
Executive Officer. Executive's duties shall include: overseeing all technology strategy and
operations; managing the engineering, data science, and cybersecurity teams; leading the
Company's AI and machine learning initiatives; and serving as a member of the Executive
Leadership Team.

2. TERM
The initial term of employment shall be three (3) years commencing on the Effective Date,
subject to earlier termination as provided herein. The Agreement shall automatically renew
for successive one-year periods unless either party provides ninety (90) days prior written
notice of non-renewal.

3. COMPENSATION

3.1 Base Salary
The Company shall pay Executive an annual base salary of USD $420,000, payable bi-weekly.
The salary shall be reviewed annually by the Compensation Committee.

3.2 Annual Bonus
Executive shall be eligible for an annual performance bonus with a target of 50% of base salary
($210,000) and a maximum of 100% of base salary ($420,000), based on achievement of objectives
mutually agreed upon by Executive and the CEO no later than January 31 of each year.

3.3 Equity
Executive shall be granted 150,000 restricted stock units ("RSUs") vesting over four (4) years
with a one-year cliff (25% after 12 months, then quarterly thereafter). Subject to acceleration
provisions in Section 7.

4. BENEFITS
Executive shall be entitled to participate in all benefit plans generally available to senior
executives, including health insurance, dental, vision, 401(k) with 6% Company match, and
an annual executive physical examination.

5. CONFIDENTIALITY AND NON-DISCLOSURE
Executive agrees to maintain in strict confidence all Confidential Information of the Company.
This obligation shall survive the termination of this Agreement indefinitely with respect to
trade secrets and for five (5) years with respect to other Confidential Information.

6. RESTRICTIVE COVENANTS

6.1 Non-Compete
For a period of twelve (12) months following termination of employment for any reason, Executive
shall not, directly or indirectly, engage in any business that competes with the Company's
financial technology products in the United States, Canada, or United Kingdom.

6.2 Non-Solicitation of Employees
For eighteen (18) months following termination, Executive shall not solicit or recruit any
employee of the Company.

6.3 Non-Solicitation of Clients
For twelve (12) months following termination, Executive shall not solicit any client of the
Company with whom Executive had material contact during the last two years of employment.

7. TERMINATION AND SEVERANCE

7.1 Termination Without Cause or Resignation for Good Reason
If the Company terminates Executive's employment without Cause, or if Executive resigns for
Good Reason (as defined in Section 7.3), Executive shall receive:
  (a) continued base salary for eighteen (18) months ("Severance Period");
  (b) a pro-rated target bonus for the year of termination;
  (c) continued health benefits through COBRA for the Severance Period;
  (d) accelerated vesting of RSUs that would have vested in the 12 months following termination.

7.2 Termination for Cause
If the Company terminates Executive's employment for Cause, Executive shall receive only accrued
and unpaid salary through the date of termination.

7.3 Definition of Good Reason
"Good Reason" means, without Executive's written consent: (a) material reduction in base salary;
(b) material diminution in duties or reporting structure; (c) required relocation of more than
fifty (50) miles; or (d) material breach of this Agreement by the Company.

7.4 Change in Control
In the event of a Change in Control (as defined herein), all unvested RSUs shall accelerate
and vest in full upon a qualifying termination within 18 months following the Change in Control
("double-trigger acceleration").

8. DISPUTE RESOLUTION
Any dispute arising under this Agreement shall be submitted to binding arbitration before JAMS
in New York, New York, under JAMS Employment Arbitration Rules.

9. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

NOVASTAR FINANCIAL GROUP, INC.         EXECUTIVE
By: _________________________           Signature: _____________________
Name: Margaret Chen                     Dr. Amara Osei
Title: Chief Executive Officer
Date: January 1, 2023                   Date: January 1, 2023
""",
    },
]


def generate_txt_files(output_dir: Path) -> None:
    """Write sample documents as plain text files."""
    for doc in SAMPLE_DOCUMENTS:
        txt_path = output_dir / doc["filename"].replace(".pdf", ".txt")
        txt_path.write_text(doc["content"], encoding="utf-8")
        print(f"  Written: {txt_path.name}")


def generate_pdf_files(output_dir: Path) -> None:
    """Write sample documents as proper PDF files using ReportLab."""
    styles = getSampleStyleSheet()
    for doc in SAMPLE_DOCUMENTS:
        pdf_path = output_dir / doc["filename"]
        document = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        story = []

        title_style = styles["Heading1"]
        body_style = styles["BodyText"]

        story.append(Paragraph(doc["title"], title_style))
        story.append(Spacer(1, 12))

        for line in doc["content"].split("\n"):
            clean = line.strip()
            if clean:
                story.append(Paragraph(clean, body_style))
            else:
                story.append(Spacer(1, 6))

        document.build(story)
        print(f"  Written: {pdf_path.name}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    print(f"Generating sample legal documents in {output_dir}/")

    if HAS_REPORTLAB:
        generate_pdf_files(output_dir)
    else:
        generate_txt_files(output_dir)

    print("Done.")
